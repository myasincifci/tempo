from lightly.models.modules import BarlowTwinsProjectionHead
from torchvision.models import resnet34, ResNet34_Weights, resnet50, ResNet50_Weights
from torch import nn
import torch

from copy import deepcopy

class Tempo34(nn.Module):
    def __init__(self) -> None:
        super(Tempo34, self).__init__()
        resnet = resnet34(ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        self.projection_head = BarlowTwinsProjectionHead(512, 1024, 1024)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

class Tempo34RGB(nn.Module):
    def __init__(self) -> None:
        super(Tempo34RGB, self).__init__()
        resnet = resnet34(ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BarlowTwinsProjectionHead(512, 1024, 1024)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

class Tempo50(nn.Module):
    def __init__(self) -> None:
        super(Tempo50, self).__init__()
        resnet = resnet50(ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # self.backbone[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        self.projection_head = BarlowTwinsProjectionHead(2048, 1024, 1024)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

# class BaselineRandomInit(nn.Module):
#     def __init__(self):
#         super(BaselineRandomInit, self).__init__()
#         resnet = resnet34()
#         self.backbone = nn.Sequential(*list(resnet.children())[:-1])
#         self.backbone[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
#         self.linear = nn.Linear(in_features=512, out_features=3, bias=True)

#     def forward(self, x):
#         x = self.backbone(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.linear(x)

#         return x

class Baseline(nn.Module):
    def __init__(self, out_features:int, freeze_backbone:bool=False, pretrain=True):
        super(Baseline, self).__init__()

        if pretrain:
            resnet = resnet34(ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet34()
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        self.linear = nn.Linear(in_features=512, out_features=out_features, bias=True)

        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return x

class BaselineRGB(nn.Module):
    def __init__(self, out_features:int, freeze_backbone:bool=False, pretrain=True):
        super(BaselineRGB, self).__init__()

        if pretrain:
            resnet = resnet34(ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet34()
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(in_features=512, out_features=out_features, bias=True)

        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return x

class LinearEval(nn.Module):
    def __init__(self, backbone: nn.Module, out_features:int, freeze_backbone:bool=False):
        super(LinearEval, self).__init__()
        self.backbone = deepcopy(backbone)
        self.linear = nn.Linear(in_features=512, out_features=out_features, bias=True)

        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return x

class LinearEvalHead(nn.Module):
    def __init__(self, out_features:int):
        super(LinearEvalHead, self).__init__()
        self.linear = nn.Linear(in_features=512, out_features=out_features, bias=True)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return x