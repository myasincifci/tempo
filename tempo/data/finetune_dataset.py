import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import torchvision.transforms as T

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

class FinetuneDataset(Dataset):
    def __init__(self, path, split_at:int, transform=None, train=True) -> None:
        self.train = train
        self.transform = transform

        total_len = len([p for p in os.listdir(path) if not p.endswith('.txt')])
        train_len = split_at

        # get annotations
        self.label_map = {}
        with open(os.path.join(path, 'annotations.txt')) as f:
            for l in f.readlines():
                a, b, l = list(map(int, l.strip().split(',')))

                for i in range(a,b):
                    self.label_map[i] = l

        self.image_paths_train = []
        self.image_paths_test = []
        for name in sorted(os.listdir(path)):
            if not name.endswith('.txt'):
                frame = int(name.split('.')[0])
                if frame in self.label_map:
                    full_path = os.path.join(path, name)
                    label = self.label_map[frame]

                    if frame >= train_len:
                        self.image_paths_test.append((full_path, label))
                    else:
                        self.image_paths_train.append((full_path, label))
        
        # self.transform = transform
        # self.image_paths = sorted([os.path.join(path, p) for p in os.listdir(path) if not p.endswith('.txt')])

        # self.total_len = len(self.image_paths)
        # self.train_len = round(self.total_len*0.8)
        # self.test_len  = round(self.total_len - self.train_len)

        # self.label_map = {}
        # with open(os.path.join(path, 'annotations.txt')) as f:
        #     for l in f.readlines():
        #         a, b, l = list(map(int, l.strip().split(',')))

        #         for i in range(a,b):
        #             self.label_map[i] = l

        # # Remove all images that are not in the label map
        # valid_image_paths_train = []
        # valid_image_paths_test = []
        # for i, _ in enumerate(self.image_paths):
        #     if i in self.label_map:

        #         valid_image_paths.append(self.image_paths[i])
        # self.valid_image_paths = sorted(valid_image_paths)

    def __len__(self) -> int:

        if self.train:
            return len(self.image_paths_train) 
        else:
            return len(self.image_paths_test)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            pth, cls = self.image_paths_train[index]
            image = Image.open(pth)
        else:
            pth, cls = self.image_paths_test[index]
            image = Image.open(pth)

        if self.transform:
            image = self.transform(image)

        return (image, cls)

if __name__ == '__main__':
    transform = T.Compose([
        T.Resize(128),
        T.ToTensor()
    ])
    
    train_dataset = FinetuneDataset('./datasets/hand_2', transform=transform, train=True)
    test_dataset = FinetuneDataset('./datasets/hand_2', transform=transform, train=False)

    a=0