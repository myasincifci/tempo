from tempo.data.tempo_dataset import TempoDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

def hand_dataset(batch_size=80, proximity=30):
    transform = T.Compose([
        T.Resize(128),
        T.ToTensor(),
        T.Grayscale()
    ])

    dataset = TempoDataset('./datasets/hand', transform=transform, proximity=proximity)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    return dataloader