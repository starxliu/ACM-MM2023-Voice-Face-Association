import torch
from torch import nn
import numpy as np

class NormalizeConv(nn.Module):
    def __init__(self,in_channel,normal_mean=[103.530,116.280,123.675],normal_std=[57.375,57.120,58.395],gray_mode=False,bgr_mode=True):
        super().__init__()
        if gray_mode:
            if bgr_mode:
                normal_mean = [(normal_mean[2]*299 + normal_mean[1]*587 + normal_mean[0]*114 + 500) / 1000]
            else:
                normal_mean = [(normal_mean[0]*299 + normal_mean[1]*587 + normal_mean[2]*114 + 500) / 1000]
        else:
            pass
        normal_mean = [-i/std for i,std in zip(normal_mean,normal_std)]
        normal_std = [1/std for std in normal_std]
        weight = np.array([[[normal_std]]])
        bias = np.array(normal_mean)
        weight = weight.transpose((3,0,1,2))
        self.conv = nn.Conv2d(in_channel,in_channel,1,stride=1,padding=0,groups=in_channel,bias=True)
        for i in self.conv.parameters():
            i.requires_grad = False
        self.conv.weight = nn.Parameter(torch.Tensor(weight),requires_grad=False)
        self.conv.bias = nn.Parameter(torch.Tensor(bias),requires_grad=False)

    def forward(self,x):
        out = self.conv(x)
        return out