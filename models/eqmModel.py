import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .vfModel import cosine_distance
import math
import os
import numpy as np
import matplotlib.pyplot as plt

class eqm(BaseModel):
    def __init__(self, args):
        super(eqm, self).__init__(args)
        num_classes = self.args.num_classes

        self.corr_num = nn.Parameter(torch.FloatTensor(num_classes), requires_grad=False)
        nn.init.constant_(self.corr_num, 0.0)
        self.compare_sum = nn.Parameter(torch.FloatTensor(num_classes), requires_grad=False)
        nn.init.constant_(self.compare_sum, 0.0)

        self.weight = nn.Parameter(torch.FloatTensor(num_classes), requires_grad=False)
        nn.init.constant_(self.weight, 1)

        self.set_metrics()

    def set_metrics(self, metrics=cosine_distance):
        self.metrics = metrics

    def forward(self, ACR, y):
        self.corr_num[y] += ACR['corr']
        self.compare_sum[y] += ACR['sum']

    def flush(self):
        '''
            Dynamic strategy
        '''
        corr_ratio = self.corr_num / self.compare_sum

        sigma = corr_ratio.std()
        miu = corr_ratio.mean() + (-1.0) * sigma
        sigma = math.pow(1, 0.5) * sigma

        Wei_distr = torch.distributions.normal.Normal(loc=miu, scale=sigma)

        nn.init.constant_(self.weight, 0.0)
        weight_sum = 0
        for i in range(self.args.num_classes):
            self.weight[i] = Wei_distr.log_prob(corr_ratio[i]).exp()
            weight_sum += self.weight[i]

        # clr
        nn.init.constant_(self.corr_num, 0.0)
        nn.init.constant_(self.compare_sum, 0.0)

    def get_weight(self, y):
        w = torch.index_select(self.weight, 0, y)
        return w
