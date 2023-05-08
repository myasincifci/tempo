import os
import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tempo.models import NewTempoLinear
from tempo.data.datasets import finetune_dataset

def test_model(model, testloader, device):

    wrongly_classified = 0
    for i, data in enumerate(testloader, 0):
        total = len(data[0])
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            preds = model(inputs).argmax(dim=1)

        wrong = (total - (preds == labels).sum()).item()
        wrongly_classified += wrong

    return 1 - (wrongly_classified / len(testloader.dataset))

def semi_sup_eval(lr, iterations, weights, train_loader, test_loader, device):
    
    model = NewTempoLinear(out_features=24, weights=None)
    model.load_state_dict(weights)
    model.linear = nn.Linear(in_features=512, out_features=24)
    model.to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses, errors, iters = [], [], []
    i = 0
    every = 1
    running_loss = 0.0
    b1 = False

    with tqdm(total=iterations, leave=False) as pbar:
        while True:
            for img, label in train_loader:            
                if i % every == 0:
                    test_error = test_model(model, test_loader, device)
                    losses.append(running_loss)
                    errors.append(test_error)
                    iters.append(i)
                    running_loss = 0

                    if i == iterations:
                        b1=True
                        break
                
                labels = nn.functional.one_hot(label, num_classes=24).float()
                inputs, labels = img.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                i += 1
                pbar.update(1)

            if i == iterations and b1:
                break
    losses, errors, iters = np.array(losses), np.array(errors), np.array(iters)

    return (losses, errors, iters)

def main(args):
    # Parse commandline-arguments
    path = args.path
    name: str = args.name
    runs: int = args.runs if args.runs else 10
    samples_pc: int = args.samples_pc
    lr: float = args.lr if args.lr else 0.01

    # Load datasets
    train_loader_ft = finetune_dataset(name='ASL-big', train=True, batch_size=128, samples_pc=samples_pc)
    test_loader_ft = finetune_dataset(train=False, batch_size=20)
    
    # Use GPU if availabel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    # Parameters for finetuning
    iterations = 200

    # Load model from path
    weights = torch.load(path)

    # Train model 
    e = []
    for i in tqdm(range(runs)):
        _, errors, iters = semi_sup_eval(lr, iterations, weights, train_loader_ft, test_loader_ft, device)
        e.append(errors.reshape(1,-1))
    e = np.concatenate(e, axis=0)
    e_mean = e.mean(axis=0)
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
    parser.add_argument('--lr', type=float, required=False)

    args = parser.parse_args()

    main(args)