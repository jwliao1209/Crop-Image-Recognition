import os
import argparse

import torch
import torch.nn as nn

from .losses import FocalLoss
from .models import EfficientNet_B0, RegNet_Y_16, ViT_L, Swin_S, Swin_V2_B, ConvNext_S, ConvNext_B
from .scheduler import WarmupCosineSchedule
from .utils import load_json


def get_device(device_id=0):
    if type(device_id) is list:
        device_id = device_id[0]

    device = torch.device(f'cuda:{device_id}'
             if torch.cuda.is_available() else 'cpu')
    
    return device


def get_train_model(model, num_classes, device_ids):
    model = get_model(model, num_classes)
    model = model.data_parallel(device_ids)

    return model


def get_model(model, num_classes):
    Model = dict(
        efficientnet_b0=EfficientNet_B0,
        regnet_y_16=RegNet_Y_16,
        vit_l=ViT_L,
        swin_s=Swin_S,
        swin_v2_b=Swin_V2_B,
        convnext_s=ConvNext_S,
        convnext_b=ConvNext_B,
    )

    return Model[model](num_classes)


def get_topk_models(args, device=None):
    models = []
    for ckpt in args.checkpoint:
        tmp_args = load_json(os.path.join('checkpoint', ckpt, 'config.json'))
        tmp_args = argparse.Namespace(**tmp_args)
        models.append(
            get_model(tmp_args.model, tmp_args.num_classes)
            )

    return models


def get_criterion(loss):
    Losses = dict(
        CE=nn.CrossEntropyLoss,
        FL=FocalLoss
        )

    return Losses[loss](label_smoothing=0.1)


def get_optimizer(args, model):
    Optimizer = dict(
        SGD=torch.optim.SGD,
        Adam=torch.optim.Adam,
        AdamW=torch.optim.AdamW
        )

    optimizer = Optimizer[args.optim](
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    return optimizer


def get_step_lr(args, optimizer):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=args.step_size // args.accum_grad_bs,
        gamma=args.gamma
        )

    return scheduler


def get_cos_annealing_lr(args, optimizer):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.epoch
        )
    
    return scheduler


def get_warm_up_cos(args, optimizer):
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=args.train_num,
        t_total= args.train_num * args.epoch)

    return scheduler


def get_scheduler(args, optimizer):
    Scheduler = dict(
        step=get_step_lr,
        cos=get_cos_annealing_lr,
        warmup_cos=get_warm_up_cos,
        cos_annealing=get_cos_annealing
        )

    return Scheduler[args.scheduler](args, optimizer)


def get_cos_annealing(args, optimizer):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        6000,
        T_mult=1,
        eta_min=0,
        last_epoch=-1,
        verbose=False)

    return scheduler
