import os
import importlib
import torch
from torch import nn
from .base_model import BaseModel
from .vfModel import Aasm
from .backBones.ResNet50 import resnet50
from .backBones.ir_se_model import IR_50, IR_152, IR_101, IR_SE_50, IR_SE_101, IR_SE_152
from .backBones.res_se_34l_model import ResNetSE34
from .subNet import vSubNet, aSubNet
from .eqmModel import eqm


def getModel(args):

    model_wrapper = args.get('model_wrapper', None)
    pretrained = args.get('weight', None)
    model_name = model_wrapper if model_wrapper != None else args.get('model', None)

    model_repo = [Aasm, vSubNet, aSubNet, eqm, resnet50, IR_50, IR_152, IR_101,
                  IR_SE_50, IR_SE_101, IR_SE_152, ResNetSE34]

    model = None
    for m in model_repo:
        if m.__name__ == model_name:
            model = m
            break

    if model is None:
        raise NotImplementedError("Model %s not found!" % (model_name))

    if model_wrapper != None:
        return model(args)
    else:
        return model(pretrained=pretrained)
