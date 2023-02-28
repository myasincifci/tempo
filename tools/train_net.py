import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from lightly.loss import BarlowTwinsLoss

from tempo.models import Tempo34RGB, BaselineRGB
from tempo.data.datasets import hand_dataset, hand_dataset_2, hand_dataset_2_ft, hand_dataset_blk, hand_dataset_blk_ft

from linear_eval import linear_eval_fast, linear_eval

from torch.utils.tensorboard import SummaryWriter

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

def train(epochs, lr, l, train_loader, pretrain, device):
    model = Tempo34RGB(pretrain=pretrain).to(device)
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
    baseline = args.baseline if args.baseline else False
    proximity = args.proximity if args.proximity else 30
    save_model = args.save_model

    print(proximity)
    train_loader = hand_dataset_2(train=True, proximity=proximity)

    train_loader_ft = hand_dataset_2_ft(train=True, subset=100)
    test_loader_ft = hand_dataset_2_ft(train=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    if baseline:
        model_bl = BaselineRGB(out_features=3, freeze_backbone=True, pretrain=True).to(device)

        e_bl = []
        for i in tqdm(range(10)):
            _, errors_bl = linear_eval_fast(500, model_bl, train_loader_ft, test_loader_ft, device)
            e_bl.append(errors_bl.reshape(1,-1))
        e_bl = np.concatenate(e_bl, axis=0).mean(axis=0)

        writer = SummaryWriter()
        for i in np.arange(len(e_bl)):
            writer.add_scalar('error', e_bl[i], i)
        writer.close()

        if save_model:
            torch.save(model_bl.state_dict(), f"model_zoo/{save_model}")

    else:
        model = train(epochs, lr, l, train_loader, pretrain=True, device=device)

        if evaluation == 'linear':

            e = []
            for i in tqdm(range(10)):
                _, errors = linear_eval_fast(500, model, train_loader_ft, test_loader_ft, device)
                e.append(errors.reshape(1,-1))
            e = np.concatenate(e, axis=0).mean(axis=0)

            writer = SummaryWriter()
            for i in np.arange(len(e)):
                writer.add_scalar('error', e[i], i)
            writer.close()

        if save_model:
            torch.save(model.state_dict(), f"model_zoo/{save_model}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--lr', type=float, required=False)
    parser.add_argument('--l', type=float, required=False)
    parser.add_argument('--eval', type=str, required=False)
    parser.add_argument('--baseline', type=bool, required=False)
    parser.add_argument('--proximity', type=int, required=False)
    parser.add_argument('--save_model', type=str, required=False)

    args = parser.parse_args()

    main(args)