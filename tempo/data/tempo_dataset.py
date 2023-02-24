import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import torchvision.transforms as T

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

class TempoDataset(Dataset):
    def __init__(self, path, transform=None, proximity:int=3, train=True) -> None:
        self.train = train
        self.p = proximity
        self.transform = transform
        self.image_paths = sorted([os.path.join(path, p) for p in os.listdir(path) if not p.endswith('.txt')])

        self.total_len = len(self.image_paths)
        self.train_len = round(self.total_len*0.8)
        self.test_len  = round(self.total_len - self.train_len)

        self.label_map = {}
        with open(os.path.join(path, 'annotations.txt')) as f:
            for l in f.readlines():
                a, b, l = list(map(int, l.strip().split(',')))

                for i in range(a,b):
                    self.label_map[i] = l

    def __len__(self) -> int:

        if self.train:
            return self.train_len 
        else:
            return self.test_len

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        offset = 0 if self.train else self.train_len
        index = index + offset

        # 1. get one element x
        image = Image.open(self.image_paths[index])

        # 2. sample element x' in the neighbourhood of x within proximity (between 1 and p where p is proximity)
        possbile_l = range(((index - self.p) if (index - self.p) >= 0 else 0), index)
        possbile_r = range(index + 1, ((index + self.p + 1) if (index + self.p + 1) <= len(self) else len(self)))
        index_d = np.random.choice(list(possbile_l) + list(possbile_r))
        image_d = Image.open(self.image_paths[index_d])

        cls = self.get_class(index)
        cls_d = self.get_class(index_d)

        if self.transform:
            image = self.transform(image)
            image_d = self.transform(image_d)

        # 3. return (x, x')
        return (image, image_d, cls, cls_d)

    def get_class(self, index):
        try:
            return self.label_map[index]
        except KeyError:
            return 3

if __name__ == '__main__':
    transform = T.Compose([
        T.Resize(128),
        T.ToTensor()
    ])
    
    train_dataset = TempoDataset('./datasets/hand_2', transform=transform, proximity=30, train=True)
    test_dataset = TempoDataset('./datasets/hand_2', transform=transform, proximity=30, train=False)

    a=0