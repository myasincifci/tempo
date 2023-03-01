import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from tempo.data.datasets import hand_dataset
from lightly.loss import BarlowTwinsLoss
from tempo.models import Tempo34, LinearEval, Baseline, LinearEvalHead

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

def test_model_fast(model, test_reps, test_dataset, device):

    wrongly_classified = 0
    for repr, label in test_reps:
        total = repr.shape[0]

        inputs,labels = repr.to(device), label.to(device)

        with torch.no_grad():
            preds = model(inputs).argmax(dim=1)

        wrong = (total - (preds == labels).sum()).item()
        wrongly_classified += wrong

    return wrongly_classified / len(test_dataset)

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

def linear_eval_fast(epochs, model, train_loader, test_loader, device):
    backbone = model.backbone
    reps = []
    for input, label in train_loader:
        repr = backbone(input.to(device)).detach()
        reps.append((repr, label.to(device)))

    test_reps = []
    for input, label in test_loader:
        repr = backbone(input.to(device)).detach()
        test_reps.append((repr, label.to(device)))

    # for repr, label in reps:
    eval_model = LinearEvalHead(out_features=10).to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(eval_model.parameters(), lr=0.01)

    losses, errors = [], []
    for epoch in range(epochs):
        running_loss = 0.0
        for repr, label in reps:
            labels = nn.functional.one_hot(label, num_classes=10).float()
            inputs, labels = repr.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = eval_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        test_error = test_model_fast(eval_model, test_reps, test_loader.dataset, device)
        losses.append(running_loss)
        errors.append(test_error)
    losses, errors = np.array(losses), np.array(errors)

    return (losses, errors)

    # plt.plot(np.arange(50), errors, '-r', label='error')
    # plt.legend(loc="upper left")
    # plt.xlabel('Epochs')
    # plt.ylabel('Test error')
    # plt.show()

    # save=True
    # if save:
    #     torch.save(eval_model, 'model_zoo/model.pth')

def linear_eval(model, train_loader, test_loader, device):
    eval_model = LinearEval(backbone=model.backbone, out_features=3, freeze_backbone=True).to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(eval_model.parameters(), lr=0.001)

    losses, errors = [], []
    for epoch in range(50):
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

    return (losses, errors)

    # plt.plot(np.arange(100), errors, '-r', label='error')
    # plt.legend(loc="upper left")
    # plt.xlabel('Epochs')
    # plt.ylabel('Test error')
    # plt.show()

    # save=True
    # if save:
    #     torch.save(eval_model, 'model_zoo/model_bl.pth')

def main():
    pass

if __name__ == '__main__':
    main()