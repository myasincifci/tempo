import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import torchvision.transforms as T

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

class FolderDataset(Dataset):
    def __init__(self, path, transform=None) -> None:
        self.transform = transform

        class_map = {'rock': 0, 'scissors': 1, 'paper': 2}

        self.image_paths = []
        for c in os.listdir(path):
            for fn in os.listdir(os.path.join(path, c)):
                self.image_paths.append((os.path.join(path, c, fn), class_map[c]))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        pth, cls = self.image_paths[index]
        image = Image.open(pth)

        if self.transform:
            image = self.transform(image)

        return (image, cls)

if __name__ == '__main__':
    transform = T.Compose([
        T.Resize(128),
        T.ToTensor()
    ])
    
    train_dataset = FolderDataset('./datasets/hand2_ft', transform=transform)

    a=0