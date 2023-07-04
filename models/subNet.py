import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
from .base_model import BaseModel
from .backBones.ir_se_model import IR_152
from .backBones.res_se_34l_model import ResNetSE34
from .backBones.ResNet50 import resnet50
from models import getModels
import os
import matplotlib.pyplot as plt

class vSubNet(BaseModel):
    def __init__(self, args, **kwargs):
        super(vSubNet, self).__init__(args)
        output_channel = self.args.output_channel
        self.model = getModels.getModel(args.pretrained)
        self.fc = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, output_channel, bias=False)
        )

    def forward(self, x, y=None):
        x = self.model(x)
        x = F.avg_pool2d(x, x.shape[-1], stride=1).view(x.shape[0], -1)
        x = self.fc(x)
        return x

class aSubNet(BaseModel):
    def __init__(self, args, **kwargs):
        super(aSubNet, self).__init__(args)
        output_channel = self.args.output_channel
        self.model = getModels.getModel(args.pretrained)
        self.fc = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, output_channel, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

