from tempo.data.time_shift_dataset import TimeShiftDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

def hand_dataset(batch_size=80, proximity=30, train=True):
    transform = T.Compose([
        T.Resize(128),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet normalization
    ])

    dataset = TimeShiftDataset('./datasets/hand', transform=transform, proximity=proximity, train=train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    return dataloader