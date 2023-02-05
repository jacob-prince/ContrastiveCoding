from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path 
from torch.hub import load_state_dict_from_url
from pdb import set_trace
    
class AlexNetGN(nn.Module):
    def __init__(self, in_channel=3, out_dim=128, l2norm=True):
        super(AlexNetGN, self).__init__()
        self._l2norm = l2norm
        conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 96, 11, 4, 2, bias=False),
            nn.GroupNorm(32, 96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        conv_block_2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 384),
            nn.ReLU(inplace=True),
        )
        conv_block_4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 384),
            nn.ReLU(inplace=True),
        )
        conv_block_5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        ave_pool = nn.AdaptiveAvgPool2d((6,6))
                
        fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        fc8 = nn.Sequential(
            nn.Linear(4096, out_dim)
        )
        head = [fc6, fc7, fc8]
        if self._l2norm: 
            head.append(Normalize(2))
        
        self.backbone = nn.Sequential(
            conv_block_1,
            conv_block_2,
            conv_block_3,
            conv_block_4,
            conv_block_5,
            ave_pool,
        )
        
        self.head = nn.Sequential(*head)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.head(x)
        return x

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class BarlowTwins(nn.Module):
    def __init__(self, backbone, lambd=0.0051, batch_size=2048, projector_sizes=[4096,4096,4096]):
        super().__init__()
        self.lambd = lambd
        self.batch_size = batch_size
        self.backbone = backbone
        self.flatten = nn.Flatten()
        
        # projector
        sizes = [256*6*6] + projector_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x):
        z = self.projector(self.flatten(self.backbone(x)))
        return self.bn(z)

def alexnet_gn_barlow_twins(pretrained=False):
    model = BarlowTwins(AlexNetGN().backbone)
    if pretrained:
        url_root = os.path.join("https://visionlab-pretrainedmodels.s3.amazonaws.com", "model_zoo", "barlow_twins")
        url = os.path.join(url_root, 'barlow_alexnet_gn_imagenet_final.pth.tar')
        print(f"... loading checkpoint: {Path(url).name}")
        checkpoint = load_state_dict_from_url(url, map_location=torch.device('cpu'))
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)        
        print("... state loaded.")
    else:
        state_dict = None

    return model, state_dict 
