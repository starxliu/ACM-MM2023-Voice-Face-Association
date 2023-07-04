import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_normal_, calculate_gain, xavier_normal_
from collections import OrderedDict
from abc import ABC, abstractmethod
from .NNutils.Normalize import NormalizeConv


class BaseModel(nn.Module,ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.get('normal_layer', False):
            self.normalize = NormalizeConv(3,normal_mean=args.normal_mean,normal_std=args.normal_std)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def freezeBN(self):
        for m in self.modules():
            class_name = m.__class__.__name__
            if class_name.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1 and not m.weight.requires_grad:
                m.eval()

    def param_groups(self, lr=None):
        params = list(filter(lambda x:x.requires_grad, self.parameters()))
        if len(params):
            if lr is not None:
                return [{'params': params, 'lr': lr}]
            else:
                return [{'params': params}]
        else:
            return []
