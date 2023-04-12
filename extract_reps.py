import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from tempo.data.datasets import finetune_dataset
from tempo.models import NewTempoLinear
from tqdm import tqdm

def tes_model(model, test_reps, test_labels, device):
    
    model.eval()

    wrongly_classified = 0
    with torch.no_grad():
        for repr, label in zip(test_reps, test_labels):
            total = repr.shape[0]

            preds = model(repr.flatten(start_dim=1)).argmax(dim=1).to('cpu')

            wrong = (total - (preds == label).sum()).item()
            wrongly_classified += wrong

    model.train()

    return 1.0 - (wrongly_classified / (test_reps.shape[0] * test_reps.shape[1]))

def get_features(reps):
    def hook(model, input, output):
        reps['reps'].append(output.data)
    return hook

def train_layer(representations_train, representations_test, labels_train, labels_test, device):
    model = nn.Linear(
            in_features = torch.tensor(representations_train.shape)[2:].prod(), 
            out_features = 24, 
            bias=True
        ).to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)

    losses, accuracies = [], []
    for epoch in tqdm(range(50)):
        for repr, label in zip(representations_train, labels_train):
            
            labels = nn.functional.one_hot(label, num_classes=24).float().to(device)
            inputs = repr.flatten(start_dim=1).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            accuracy = tes_model(model, representations_test, labels_test, device)
            accuracies.append(accuracy)

    
    return accuracies

def eval_layer(layer, model, train_loader, test_loader , device):
    
    reps_train = {
        'reps': [],
        'labels': []
    }

    reps_test = {
        'reps': [],
        'labels': []
    }

    h1 = layer.register_forward_hook(get_features(reps_train))

    # Compute train representations

    with torch.no_grad():
        for input, label in tqdm(train_loader):

            _ = model.backbone(input.to(device)).detach()
            reps_train['labels'].append(label)

    reps_train['reps'] = torch.stack(reps_train['reps'])
    reps_train['labels'] = torch.stack(reps_train['labels'])

    h1.remove()

    h2 = layer.register_forward_hook(get_features(reps_test))

    # Compute test representations

    with torch.no_grad():
        for input, label in tqdm(test_loader):

            _ = model.backbone(input.to(device)).detach()
            reps_test['labels'].append(label)

    reps_test['reps'] = torch.stack(reps_test['reps'])
    reps_test['labels'] = torch.stack(reps_test['labels'])

    h2.remove()

    representations_train = reps_train['reps']
    representations_test  = reps_test['reps']

    labels_train = reps_train['labels']
    labels_test  = reps_test['labels']

    accuracies = train_layer(representations_train, representations_test, labels_train, labels_test, device)

    return accuracies

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    train_loader_ft = finetune_dataset(name='ASL-big', train=True, batch_size=20)
    test_loader_ft = finetune_dataset(train=False, batch_size=20)

    weights = torch.load("./model_zoo/baseline.pth")
    model = NewTempoLinear(out_features=24, weights=None, freeze_backbone=True)
    model.load_state_dict(weights)
    model.to(device)

    layers = [
        model.backbone[4], # layer1
        model.backbone[5], # layer2
        model.backbone[6], # layer3
        model.backbone[7], # layer4
        model.backbone[8]  # adaptive pooling
    ]
    accuracies = []
    for layer in layers:
        accuracy = eval_layer(layer, model, train_loader_ft, test_loader_ft, device)
        accuracies.append(accuracy)

    maxes = torch.tensor(accuracies).max(dim=1)[0]
    print(maxes)
    
    weights = torch.load("./model_zoo/asl_big_e10_p30_run5.pth")
    model = NewTempoLinear(out_features=24, weights=None, freeze_backbone=True)
    model.load_state_dict(weights)
    model.to(device)

    layers = [
        model.backbone[4], # layer1
        model.backbone[5], # layer2
        model.backbone[6], # layer3
        model.backbone[7], # layer4
        model.backbone[8]  # adaptive pooling
    ]
    accuracies = []
    for layer in layers:
        accuracy = eval_layer(layer, model, train_loader_ft, test_loader_ft, device)
        accuracies.append(accuracy)

    maxes_tp = torch.tensor(accuracies).max(dim=1)[0]
    print(maxes)

    fig, ax = plt.subplots()

    ax.plot(maxes, '-ro', label='Baseline')
    ax.plot(maxes_tp, '-bo', label='Tempo')
    ax.legend()
    ax.set_xticks([0,1,2,3,4])

    fig.canvas.draw()
    ax.set_xticklabels(['layer1', 'layer2', 'layer3', 'layer4', 'adaptive-pooling'])
    plt.show()

if __name__ == '__main__':
    main()