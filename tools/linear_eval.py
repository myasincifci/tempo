import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter


from tempo.data.datasets import finetune_dataset
from tempo.models import NewTempoLinear


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

def linear_eval_new(epochs, model, train_loader, test_loader, device):

    model.linear = nn.Linear(in_features=512, out_features=24, bias=True).to(device) # Fresh detection head

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
    optimizer = torch.optim.SGD(model.linear.parameters(), lr=0.01)

    losses, errors = [], []
    for epoch in range(epochs):
        running_loss = 0.0
        for repr, label in reps:
            labels = nn.functional.one_hot(label, num_classes=24).float()
            inputs, labels = repr.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.linear(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        test_error = test_model_fast(model.linear, test_reps, test_loader.dataset, device)
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
    num_epochs = 100

    # Load model from path
    weights = torch.load(path)
    model = NewTempoLinear(out_features=10, weights=None,freeze_backbone=True)
    model.load_state_dict(weights)
    model.to(device)

    # Train model 
    e = []
    for i in tqdm(range(num_runs)):
        _, errors = linear_eval_new(num_epochs, model, train_loader_ft, test_loader_ft, device)
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