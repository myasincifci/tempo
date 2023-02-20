from tempo.data.time_shift_dataset import TimeShiftDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import numpy as np

def hand_dataset(batch_size:int=80, proximity:int=30, train:bool=True, subset:int=None):
    transform = T.Compose([
        T.Resize(128),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet normalization
    ])

    dataset = TimeShiftDataset('./datasets/hand', transform=transform, proximity=proximity, train=train)

    if subset:
        ss_indices = np.random.choice(len(dataset), subset, replace=False)
        dataset = torch.utils.data.Subset(dataset, ss_indices)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    return dataloader