from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

from tempo.models import NewTempoLinear

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
    
    model = NewTempoLinear(out_features=10, weights=weights).to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    losses, errors = [], []
    for epoch in range(epochs):
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

def main():
    pass

if __name__ == '__main__':
    main()