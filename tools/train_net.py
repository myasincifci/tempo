import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from lightly.loss import BarlowTwinsLoss

from tempo.models import Tempo34RGB, BaselineRGB
from tempo.data.datasets import hand_dataset

from linear_eval import linear_eval_fast, linear_eval

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    losses = []
    for image, image_d, _, _ in tqdm(dataloader):
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

def train(epochs, lr, l, train_loader, device):
    model = Tempo34RGB().to(device)
    criterion = BarlowTwinsLoss(lambda_param=l)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001)

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

    return model

def main(args):
    epochs = args.epochs if args.epochs else 5
    lr = args.lr if args.lr else 1e-3
    l = args.l if args.l else 1e-3
    evaluation = args.eval if args.eval else 'linear'
    comp_bl = args.comp_bl if args.comp_bl else True

    train_loader = hand_dataset(train=True)
    test_loader = hand_dataset(train=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    # if evaluation == 'baseline':
    #     print('Evaluating baseline ...')
    #     model = BaselineRGB(out_features=3, freeze_backbone=True, pretrain=True).to(device)
    #     linear_eval(model, train_loader, test_loader, device)
    # else:
    #     model = train(epochs, lr, l, train_loader, device)
    #     if evaluation == 'linear':
    #         linear_eval(model, train_loader, test_loader, device)

    model = train(epochs, lr, l, train_loader, device)

    if evaluation == 'linear':

        e = []
        for i in tqdm(range(30)):
            _, errors = linear_eval_fast(100, model, train_loader, test_loader, device)
            e.append(errors.reshape(1,-1))
        e = np.concatenate(e, axis=0).mean(axis=0)

        plt.plot(np.arange(len(e)), e, '-b', label='error_tempo')

    if comp_bl:

        model_bl = BaselineRGB(out_features=3, freeze_backbone=True, pretrain=True).to(device)

        e_bl = []
        for i in tqdm(range(30)):
            _, errors_bl = linear_eval_fast(100, model_bl, train_loader, test_loader, device)
            e_bl.append(errors_bl.reshape(1,-1))
        e_bl = np.concatenate(e_bl, axis=0).mean(axis=0)

        plt.plot(np.arange(len(e_bl)), e_bl, '-r', label='error_bl')

    
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Test error')
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--lr', type=float, required=False)
    parser.add_argument('--l', type=float, required=False)
    parser.add_argument('--eval', type=str, required=False)
    parser.add_argument('--comp_bl', type=bool, required=False)

    args = parser.parse_args()

    main(args)