from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

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

    model.linear = nn.Linear(in_features=512, out_features=10, bias=True).to(device) # Fresh detection head

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
            labels = nn.functional.one_hot(label, num_classes=10).float()
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

def main():
    pass

if __name__ == '__main__':
    main()