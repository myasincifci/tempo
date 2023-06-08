from tempo.data.time_shift_dataset import TimeShiftDataset
# from tempo.data.tempo_dataset import TempoDataset
from tempo.data.tempo_dataset_new import TempoDataset
from tempo.data.finetune_dataset import FinetuneDataset
from tempo.data.dataset import Dataset

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

import numpy as np

transform = T.Compose([
    T.Resize(128),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform2 = T.Compose([
    T.Resize(128),
    T.ToTensor(),
    # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform50 = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def video_dataset50(batch_size=80, proximity=30):
    dataset = TempoDataset('./datasets/ASL-big/frames', transform=transform50, proximity=proximity)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    print(50)

    return dataloader

def finetune_dataset50(name='ASL-big', batch_size=80, train=True, samples_pc=None):
    dataset = Dataset(f'./datasets/{name}', transform=transform50, train=train, samples_pc=samples_pc)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    print(50)

    return dataloader

def video_dataset(batch_size=80, proximity=30):
    dataset = TempoDataset('./datasets/ASL-big/frames', transform=transform, proximity=proximity)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)

    return dataloader

def finetune_dataset(name='ASL-big', batch_size=80, train=True, samples_pc=None, drop_last=False):
    dataset = Dataset(f'./datasets/{name}', transform=transform, train=train, samples_pc=samples_pc)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=2)

    return dataloader

def finetune_dataset2(name='ASL-big', batch_size=80, train=True, samples_pc=None):
    dataset = Dataset(f'../datasets/{name}', transform=transform2, train=train, samples_pc=samples_pc)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)

    return dataloader

################################################################################

def hand_dataset(batch_size=80, proximity=30, train=True):
    dataset = TimeShiftDataset('./datasets/hand', transform=transform, proximity=proximity, train=train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    return dataloader

def hand_dataset_2_ft(batch_size=80, train=True, subset=None):
    dataset = FinetuneDataset('./datasets/hand_2', split_at=6617, transform=transform, train=train)

    if subset:
        ss_indices = np.random.choice(len(dataset), subset, replace=False)
        dataset = torch.utils.data.Subset(dataset, ss_indices)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    return dataloader

def hand_dataset_blk(batch_size=80, proximity=30, train=True):
    dataset = TempoDataset('./datasets/hand_blk', transform=transform, proximity=proximity, train=train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    return dataloader

def hand_dataset_blk_ft(batch_size=80, train=True, subset=None):
    dataset = FinetuneDataset('./datasets/hand_blk',split_at=1389, transform=transform, train=train)

    if subset:
        ss_indices = np.random.choice(len(dataset), subset, replace=False)
        dataset = torch.utils.data.Subset(dataset, ss_indices)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    return dataloader