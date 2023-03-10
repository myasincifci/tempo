import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from lightly.loss import BarlowTwinsLoss
from torchvision.models import ResNet34_Weights

from tempo.models import Tempo34RGB, NewBaseline, NewTempoLinear, get_resnet_weights
from tempo.data.datasets import video_dataset, finetune_dataset

from linear_eval import linear_eval_new
from semi_sup_eval import semi_sup_eval

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

    return model.backbone.state_dict()

def main(args):
    # Parse commandline-arguments
    epochs = args.epochs if args.epochs else 1
    lr = args.lr if args.lr else 1e-3
    l = args.l if args.l else 1e-3
    evaluation = args.eval if args.eval else 'finetune'
    baseline = args.baseline if args.baseline else False
    proximity = args.proximity if args.proximity else 30
    save_model = args.save_model

    # Load datasets
    train_loader = video_dataset(proximity=proximity)
    train_loader_ft = finetune_dataset(name='ASL-big', train=True, batch_size=10)
    test_loader_ft = finetune_dataset(train=False, batch_size=10)

    # Use GPU if availabel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    # Parameters for finetuning
    num_runs = 10
    num_epochs = 100

    # Choose model
    if baseline:
        model = NewBaseline(out_features=10, pretrain=True).to(device)
    else:
        weights = train(epochs, lr, l, train_loader, pretrain=True, device=device)
        model = NewTempoLinear(weights, out_features=24).to(device)

    # Train model 
    if evaluation == 'linear':
        e = []
        for i in tqdm(range(num_runs)):
            _, errors = linear_eval_new(num_epochs, model, train_loader_ft, test_loader_ft, device)
            e.append(errors.reshape(1,-1))
        e = np.concatenate(e, axis=0).mean(axis=0)
    
    elif evaluation == 'finetune':
        e = []
        for i in tqdm(range(num_runs)):
            _, errors = semi_sup_eval(num_epochs, weights, train_loader_ft, test_loader_ft, device)
            e.append(errors.reshape(1,-1))
        e = np.concatenate(e, axis=0).mean(axis=0)
    
    else:
        e = []

    # Write to tensorboard
    writer = SummaryWriter()
    for i in np.arange(len(e)):
        writer.add_scalar('error', e[i], i)
    writer.close()

    # Save model weights
    if save_model:
        torch.save(model.state_dict(), f"model_zoo/{save_model}") # TODO: cant save models evaluated with finetune because only passing weights

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