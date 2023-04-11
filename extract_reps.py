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

def get_features(reps, layer):
    def hook(model, input, output):
        reps[layer].append(output.data)
    return hook

def train_layer(representations_train, representations_test, labels_train, labels_test, device):
    model = nn.Linear(
            in_features = torch.tensor(representations_train.shape)[2:].prod(), 
            out_features = 24, 
            bias=True
        ).to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001)

    losses, accuracies = [], []
    for epoch in range(50):
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    train_loader_ft = finetune_dataset(name='ASL-big', train=True, batch_size=20)
    test_loader_ft = finetune_dataset(train=False, batch_size=20)

    weights = torch.load("./model_zoo/baseline.pth")
    model = NewTempoLinear(out_features=24, weights=None, freeze_backbone=True)
    model.load_state_dict(weights)
    model.to(device)

    reps_train = {
        'l1': [],
        'l2': [],
        'l3': [],
        'l4': [],
        'labels': []
    }

    reps_test = {
        'l1': [],
        'l2': [],
        'l3': [],
        'l4': [],
        'labels': []
    }

    h1 = model.backbone[4].register_forward_hook(get_features(reps_train, 'l1'))
    h2 = model.backbone[5].register_forward_hook(get_features(reps_train, 'l2'))
    h3 = model.backbone[6].register_forward_hook(get_features(reps_train, 'l3'))
    h4 = model.backbone.register_forward_hook(get_features(reps_train, 'l4'))

    # Compute train representations

    with torch.no_grad():
        for input, label in tqdm(train_loader_ft):

            _ = model.backbone(input.to(device)).detach()
            reps_train['labels'].append(label)

    reps_train['l1'] = torch.stack(reps_train['l1'])
    reps_train['l2'] = torch.stack(reps_train['l2'])
    reps_train['l3'] = torch.stack(reps_train['l3'])
    reps_train['l4'] = torch.stack(reps_train['l4'])
    reps_train['labels'] = torch.stack(reps_train['labels'])

    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()

    h1 = model.backbone[4].register_forward_hook(get_features(reps_test, 'l1'))
    h2 = model.backbone[5].register_forward_hook(get_features(reps_test, 'l2'))
    h3 = model.backbone[6].register_forward_hook(get_features(reps_test, 'l3'))
    h4 = model.backbone.register_forward_hook(get_features(reps_test, 'l4'))

    # Compute test representations

    with torch.no_grad():
        for input, label in tqdm(test_loader_ft):

            _ = model.backbone(input.to(device)).detach()
            reps_test['labels'].append(label)

    reps_test['l1'] = torch.stack(reps_test['l1'])
    reps_test['l2'] = torch.stack(reps_test['l2'])
    reps_test['l3'] = torch.stack(reps_test['l3'])
    reps_test['l4'] = torch.stack(reps_test['l4'])
    reps_test['labels'] = torch.stack(reps_test['labels'])

    ############################################################################

    representations_train = reps_train['l3']
    representations_test  = reps_test['l3']

    labels_train = reps_train['labels']
    labels_test  = reps_test['labels']

    plt.show()


if __name__ == '__main__':
    main()