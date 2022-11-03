import os
import argparse
import torch
import torch.nn as nn

from .losses import FocalLoss
from .models import EfficientNet_B0, RegNet_Y_16, Swin_S, ConvNext_S
from .utils import load_json


def get_device(device_id=0):
    device = torch.device(f'cuda:{device_id}'
             if torch.cuda.is_available() else 'cpu')
    
    return device


def get_model(model, num_classes):
    Model = {
        'efficientnet_b0': EfficientNet_B0,
        'regnet_y_16':     RegNet_Y_16,
        'swin_s':          Swin_S,
        'convnext_s':      ConvNext_S,
    }
    return Model[model](num_classes)


def get_topk_models(args, device):
    models = []
    for ckpt in args.checkpoint:
        tmp_args = load_json(os.path.join('checkpoint', ckpt, 'config.json'))
        tmp_args = argparse.Namespace(**tmp_args)
        models.append(
            get_model(tmp_args.model, tmp_args.num_classes)
            )

    return models


def get_criterion(loss):
    Losses = {
        'CE':   nn.CrossEntropyLoss,
        'FL':   FocalLoss,
    }
    return Losses[loss]()


def get_optimizer(args, model):
    Optimizer = {
        'SGD':   torch.optim.SGD,
        'Adam':  torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
    }
    optimizer = Optimizer[args.optim](
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    return optimizer


def get_scheduler(args, optimizer):
    Scheduler = {
        'step': torch.optim.lr_scheduler.StepLR(
              optimizer=optimizer,
              step_size=args.step_size // args.accumulate_grad_bs,
              gamma=args.gamma
        ),
        'cos': torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.epoch
        )
    }
    return Scheduler[args.scheduler]
