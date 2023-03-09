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

    return wrongly_classified / len(testloader.dataset)

def semi_sup_eval(epochs, weights, train_loader, test_loader, device):
    
    model = NewTempoLinear(out_features=10, weights=None)
    model.load_state_dict(weights)
    model.linear = nn.Linear(in_features=512, out_features=10)
    model.to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    losses, errors = [], []
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for img, label in train_loader:
            labels = nn.functional.one_hot(label, num_classes=10).float()
            inputs, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        test_error = test_model(model, test_loader, device)
        losses.append(running_loss)
        errors.append(test_error)
    losses, errors = np.array(losses), np.array(errors)

    return (losses, errors)

def main(args):
    # Parse commandline-arguments
    path = args.path

    # Load datasets
    train_loader_ft = finetune_dataset(name='asl_finetune_20', train=True, batch_size=10)
    test_loader_ft = finetune_dataset(train=False, batch_size=10)

    # Use GPU if availabel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    # Parameters for finetuning
    num_runs = 10
    num_epochs = 150

    # Load model from path
    weights = torch.load(path)

    # Train model 
    e = []
    for i in tqdm(range(num_runs)):
        _, errors = semi_sup_eval(num_epochs, weights, train_loader_ft, test_loader_ft, device)
        e.append(errors.reshape(1,-1))
    e = np.concatenate(e, axis=0).mean(axis=0)

    # Write to tensorboard
    writer = SummaryWriter()
    for i in np.arange(len(e)):
        writer.add_scalar('error', e[i], i)
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=False)

    args = parser.parse_args()

    main(args)