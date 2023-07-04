import os
import torch
from torch.utils.data import DataLoader
import argparse
from dataloaders.myDataset import MyDataset
import yaml
from easydict import EasyDict as edict
import copy
import sys
import numpy as np
import random
from models.getModels import getModel
from utils.utils import load_pretrained_model, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, metavar='CFG', help='config file')

class BaseContainer(object):
    def __init__(self):
        args = parser.parse_args()
        fi = open(args.cfg,'r')
        args = yaml.load(fi, Loader=yaml.FullLoader)
        self.args = edict(args)
        # self.Dataset_train = MyDataset
        self.Dataset_val = MyDataset
        self.args.training.cuda = not self.args.training.get('no_cuda',False)
        self.args.training.gpus = torch.cuda.device_count()

        torch.backends.cudnn.benchmark = True
        torch.manual_seed(1)

    def init_evaluation_container(self):
        self.model = getModel(self.args.models)
        state_dict, _, _, _ = load_checkpoint(checkpoint_path=self.args.evaluation.trained_model)
        load_pretrained_model(self.model, state_dict)
        self.model = self.model.cuda()

    def gen_optimizer(self, train_params, stage=0):
        args = self.args.training.optimizer
        self.optimizer = dict()
        for name in args.keys():
            item = args[name]
            params = []
            for i in item.train_params:
                params += train_params[i]
            if len(params) == 0:
                continue
            if item.optim_method == 'sgd':
                self.optimizer[name] = torch.optim.SGD(
                    params,
                    momentum=item.get('momentum', 0.0),
                    lr=item.lr * item.get('lr_decay', 1) ** stage,
                    weight_decay=item.get('weight_decay', 0),
                    nesterov=item.get('nesterov', False)
                )
            elif item.optim_method == 'adagrad':
                self.optimizer[name] = torch.optim.Adagrad(
                    params,
                    lr=item.lr * item.get('lr_decay', 1) ** stage,
                    weight_decay=item.get('weight_decay', 0),
                )
            elif item.optim_method == 'adam':
                self.optimizer[name] = torch.optim.Adam(
                    params,
                    lr=item.lr * item.get('lr_decay', 1) ** stage,
                    weight_decay=item.get('weight_decay', 0),
                    betas=item.get('betas', (0.9, 0.999))
                )
            else:
                raiseNotImplementedError(
                    "optimizer %s not implemented!"%item.optim_method)

    def training(self):
        pass

    def validation(self):
        pass
