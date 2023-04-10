import os
import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


from tempo.data.datasets import finetune_dataset
from tempo.models import NewTempoLinear


def test_model_fast(model, test_reps, test_dataset, device):

    model.eval()

    wrongly_classified = 0
    for repr, label in test_reps:
        total = repr.shape[0]

        inputs, labels = repr.to(device), label.to(device)

        with torch.no_grad():
            preds = model(inputs).argmax(dim=1)

        wrong = (total - (preds == labels).sum()).item()
        wrongly_classified += wrong

    model.train()

    return 1.0 - (wrongly_classified / len(test_dataset))


def linear_eval_new(iterations, model, train_loader, test_loader, device):

    model.linear = nn.Linear(in_features=512, out_features=24, bias=True).to(
        device)  # Fresh detection head

    reps = []
    test_reps = []
    with torch.no_grad():
        for input, label in train_loader:
            repr = model.backbone(input.to(device)).detach()
            repr = torch.flatten(repr, start_dim=1)
            reps.append((repr, label.to(device)))

        for input, label in test_loader:
            repr = model.backbone(input.to(device)).detach()
            repr = torch.flatten(repr, start_dim=1)
            test_reps.append((repr, label.to(device)))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.linear.parameters(), lr=0.01, weight_decay=0.0001)

    losses, errors, iters_ = [], [], []
    i = 0
    every = 1
    running_loss = 0.0
    b1 = False
    while True:
        for repr, label in reps:
            if i % every == 0:
                test_error = test_model_fast(model.linear, test_reps, test_loader.dataset, device)
                losses.append(running_loss)
                errors.append(test_error)
                iters_.append(i)
                running_loss = 0

                if i == iterations:
                    b1=True
                    break

            labels = nn.functional.one_hot(label, num_classes=24).float()
            inputs, labels = repr.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.linear(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            i += 1

        if i == iterations and b1:
            break
    losses, errors, iters_ = np.array(losses), np.array(errors), np.array(iters_)

    return (losses, errors, iters_)

def main(args):
    # Parse commandline-arguments
    path: str = args.path
    name: str = args.name
    runs: int = args.runs if args.runs else 10
    samples_pc: int = args.samples_pc

    # Load datasets
    train_loader_ft = finetune_dataset(name='ASL-big', train=True, batch_size=10, samples_pc=samples_pc)
    test_loader_ft = finetune_dataset(train=False, batch_size=10)

    # Use GPU if availabel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    # Parameters for finetuning
    iterations = 3000

    # Load model from path
    weights = torch.load(path)
    model = NewTempoLinear(out_features=24, weights=None, freeze_backbone=True)
    model.load_state_dict(weights)
    model.to(device)

    # Train model
    e = []
    iters = None
    for i in tqdm(range(runs)):
        _, errors, iters = linear_eval_new(
            iterations, model, train_loader_ft, test_loader_ft, device)
        e.append(errors.reshape(1, -1))
    
    e = np.concatenate(e, axis=0)
    e_mean = e.mean(axis=0)
    print(e_mean.shape)
    print(iters.shape)
    e_std = e.std(axis=0)

    # Write to tensorboard
    log_dir = os.path.join("runs", name, str(samples_pc)) if name else None
    writer = SummaryWriter(log_dir)
    for i in np.arange(len(e_mean)):
        writer.add_scalar('accuracy', e_mean[i], iters[i])
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=False)
    parser.add_argument('--runs', type=int, required=False)
    parser.add_argument('--name', type=str, required=False)
    parser.add_argument('--samples_pc', type=int, required=False)

    args = parser.parse_args()

    main(args)
