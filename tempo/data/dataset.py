import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import torchvision.transforms as T

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

class Dataset(Dataset):
    def __init__(self, path, transform=None, train=True) -> None:
        self.transform = transform

        class_map = {'rock': 0, 'paper': 1, 'scissors': 2}

        self.image_paths = []
        split = 'train' if train else 'test'
        
        for c in os.listdir(os.path.join(path, split)):
            for name in os.listdir(os.path.join(path, split,c)):
                p = os.path.join(path, split, c, name)
                self.image_paths.append((p, int(c)))

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
    
    dataset = Dataset('./datasets/finetune', transform=transform, train=False)

    a=0