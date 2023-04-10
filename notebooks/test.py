import torch
import matplotlib.pyplot as plt
import numpy as np
from tempo.data.datasets import finetune_dataset
from tempo.models import NewTempoLinear
from linear_eval import linear_eval_new

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    train_loader_ft = finetune_dataset(name='ASL-big', train=True, batch_size=10)
    test_loader_ft = finetune_dataset(train=False, batch_size=10)

    weights = torch.load('model_zoo/baseline.pth')
    model = NewTempoLinear(out_features=24, weights=None, freeze_backbone=True)
    model.load_state_dict(weights)
    _ = model.to(device)

    losses, errors, iters = linear_eval_new(3_000, model, train_loader_ft, test_loader_ft, device)

    plt.plot(iters, errors)
    plt.show()

if __name__ == '__main__':
    main()