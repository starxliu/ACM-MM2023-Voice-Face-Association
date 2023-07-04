import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel
# from models.getModels import getModel
from models import getModels
import copy
import torchvision.models as models
import matplotlib.pyplot as plt
import os

class Aasm(BaseModel):

    def __init__(self, args, **kwargs):
        super(Aasm, self).__init__(args)

        if not args.get('Arch', False):
            raise RuntimeError("Model initialization failed!")

        if not (args.Arch.get('vNet', False) and args.Arch.get('aNet', False) and args.Arch.get('eqm', False)):
            raise RuntimeError("The model components are incomplete!")

        self.vNet = getModels.getModel(args.Arch.vNet)
        self.aNet = getModels.getModel(args.Arch.aNet)
        self.eqm = getModels.getModel(args.Arch.eqm)

        for cpt in args.Arch.keys():
            model = getattr(self, cpt)
            if model.args.get('freeze', False):
                for p in model.parameters():
                    p.requires_grad = False

        self.corr_num = nn.Parameter(torch.FloatTensor(924), requires_grad=False)
        nn.init.constant_(self.corr_num, 0.0)
        self.compare_sum = nn.Parameter(torch.FloatTensor(924), requires_grad=False)
        nn.init.constant_(self.compare_sum, 0.0)

# def flushWeight(self):
#     self.eqm.flush()

def Euclidean_distance(x, y):
    return ((x - y)**2).sum(-1)

def cosine_distance(x, y):
    return (F.normalize(x) * F.normalize(y)).sum(-1)