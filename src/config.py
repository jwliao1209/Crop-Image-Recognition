import torch
import torch.nn as nn

from src.losses import FL, MCCE
from src.optimizer import ranger21

# from src.models.swin import SwinTransformer, load_swin_pretrained
# from src.models.cswin import CSWin_96_24322_base_384, load_cswin_checkpoint
# from src.models.convnext import ConvNeXt_B
from src.models.efficientnet_b4 import EfficientNetB4


def get_model(args):
    Model = {
        #'ConvB': ConvNeXt_B,
        #'Swin': SwinTransformer,
        #'CSwin': CSWin_96_24322_base_384,
        'EfficientB4': EfficientNetB4
    }
    model = Model[args.model](num_classes=args.num_classes)

    if args.pretrain and args.model in ['Swin', 'CSwin']:
        Weight = {
            'Swin': load_swin_pretrained,
            'CSwin': load_cswin_checkpoint
        }
        model = Weight[args.model](model)
    elif args.pretrain and args.model in ['EfficientB4']:
        model = EfficientNetB4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1', num_classes=args.num_classes)

    return model


def get_criterion(args, device):
    Losses = {
        'CE':   nn.CrossEntropyLoss,
        'MCCE': MCCE.MCCE_Loss,
        'FL':   FL.FocalLoss,
        'FLSD': FL.FocalLossAdaptive
    }
    criterion = Losses[args.loss]()

    return criterion


def get_optimizer(args, model):
    Optimizer = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'Ranger': ranger21.Ranger21
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
              step_size=args.step_size,
              gamma=args.gamma
        ),
        'cos': torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.epoch
        )
    }
    scheduler = Scheduler[args.scheduler]

    return scheduler
