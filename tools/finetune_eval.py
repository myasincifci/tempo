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


def ft_eval(epochs, model, train_loader, test_loader, device):
    eval_model = LinearEval(backbone=model.backbone, out_features=3, freeze_backbone=False).to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(eval_model.parameters(), lr=0.001)

    i = 0
    iterations, losses, errors = [], [], []
    for epoch in range(epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, _, labels, _ = data
            labels = nn.functional.one_hot(labels, num_classes=3).float()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = eval_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            i += 1

        test_error = test_model(eval_model, test_loader.dataset, test_loader, device)
        losses.append(running_loss)
        errors.append(test_error)
        iterations.append(i)
    iterations, losses, errors = np.array(iterations), np.array(losses), np.array(errors)

    return (iterations, losses, errors)

def main():
    pass

if __name__ == '__main__':
    main()