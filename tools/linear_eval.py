import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from tempo.data.datasets import hand_dataset
from lightly.loss import BarlowTwinsLoss
from tempo.models import Tempo34, LinearEval, BaselineImagenet1K

import matplotlib.pyplot as plt
import numpy as np

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    losses = []
    for image, image_d in tqdm(dataloader):
        image = image.to(device)
        image_d = image_d.to(device)
        
        z0 = model(image)
        z1 = model(image_d)
        loss = criterion(z0, z1)
        losses.append(loss.detach())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = torch.tensor(losses).mean()
    return avg_loss

def train(epochs, lr, l):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    dataloader = hand_dataset()

    model = TimeShiftModel().to(device)
    criterion = BarlowTwinsLoss(lambda_param=l)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001)

    for epoch in range(epochs):
        train_one_epoch(model, dataloader, criterion, optimizer, device)

def test_model(model, test_dataset, testloader, device):

    wrongly_classified = 0
    for i, data in enumerate(testloader, 0):
        total = len(data[0])
        inputs, _, labels, _ = data
        inputs,labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            preds = model(inputs).argmax(dim=1)

        wrong = (total - (preds == labels).sum()).item()
        wrongly_classified += wrong

    return wrongly_classified / len(test_dataset)

def linear_eval(model, train_loader, test_loader, device):
    # eval_model = LinearEval(backbone=model.backbone, out_features=3, freeze_backbone=True).to(device)
    eval_model = BaselineImagenet1K(out_features=3, freeze_backbone=True).to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(eval_model.parameters(), lr=0.001)

    losses, errors = [], []
    for epoch in range(30):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, _, labels, _ = data
            labels = nn.functional.one_hot(labels, num_classes=3).float()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = eval_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        test_error = test_model(eval_model, test_loader.dataset, test_loader, device)
        losses.append(running_loss)
        errors.append(test_error)
    losses, errors = np.array(losses), np.array(errors)

    plt.plot(np.arange(30), errors, '-r', label='error')
    plt.legend(loc="upper left")
    plt.xlabel('Epochs')
    plt.ylabel('Test error')
    plt.show()

def main(args):
    pass

if __name__ == '__main__':
    main()