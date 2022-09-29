import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from src.logger import Logger
from src.dataset import get_train_val_loader
from src.trainer import train_step, val_step
from src.config import *
from src.utils import *


def train(args):
    init_random_seed(args.seed)
    save_dir = os.path.join(args.save_root, get_time())
    weight_dir = os.path.join(save_dir, 'weight')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    save_json(args, os.path.join(save_dir, 'config.json'))
    log = Logger(os.path.join(save_dir, 'result.csv'))
    device = torch.device(f'cuda:{args.device[0]}'
                          if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_train_val_loader(args)
    model = get_model(args)
    model = nn.DataParallel(model, device_ids=args.device)
    model.to(device)
    criterion = get_criterion(args, device)
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_scheduler(args, optimizer)

    if args.loss == 'MCCE':
        use_mc_loss = True

    else:
        use_mc_loss = False

    # Training
    for ep in range(1, args.epoch+1):
        # training step
        train_record, lr_scheduler = train_step(
            ep, model, train_loader, criterion, use_mc_loss, optimizer, device, lr_scheduler)
        log.add(
            epoch=ep,
            type='train',
            **train_record,
            lr=lr_scheduler.get_last_lr()[0]
        )

        # validation step
        val_record = val_step(ep, model, val_loader, criterion, use_mc_loss, device)
        log.add(
            epoch=ep,
            type='val',
            **val_record,
            lr=lr_scheduler.get_last_lr()[0]
        )
        log.save()
        save_topk_ckpt(model, ep, val_record['acc'], weight_dir, topk=5)
        # lr_scheduler.step()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', type=int, default=0,
                        help='folder default fold_0.csv')
    parser.add_argument('-bs', '--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--train_num', type=int, default=71608,
                        help='default take all train data to training')
    parser.add_argument('--valid_num', type=int, default=8954,
                        help='default take all valid data to training')
    parser.add_argument('-ep', '--epoch', type=int, default=100,
                        help='epochs')
    parser.add_argument('-cls', '--num_classes', type=int, default=33,
                        help='number of classes')
    parser.add_argument('--model', type=str, default='EfficientB4',
                        help='model')
    parser.add_argument('--pretrain', type=bool, default=0,
                        help='pretrained weight')
    parser.add_argument('--img_size', type=int, default=384,
                        help='crop and resize to img_size')
    parser.add_argument('--prob_noise', type=float, default=0.01,
                        help='random gaussian noise prob')    

    # set optimization
    parser.add_argument('--loss', type=str, default='CE',
                        help='loss function')
    parser.add_argument('--optim', type=str, default='AdamW',
                        help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-3,
                         help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--scheduler', type=str, default='step',
                        help='learning rate schedule')
    parser.add_argument('--step_size', type=int, default=3,
                        help='learning rate decay period')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='learning rate decay factor')

    # augmentation
    parser.add_argument('--autoaugment', type=bool, default=1,
                        help='autoaugmentation')
    parser.add_argument('--rot_degree', type=float, default=10,
                        help='degree of rotation')
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help='Probabiliy of horizontal flip')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='Probabiliy of adding gaussian noise')

    # set device
    parser.add_argument('--num_workers', type=int, default=4,
                        help='numbers of workers')
    parser.add_argument('--device', type=int, default=[0], nargs='+',
                        help='index of gpu device')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    # set path
    parser.add_argument('--root', type=str, default='./dataset',
                        help='dataset root')
    parser.add_argument('--save_root', type=str, default='./checkpoint',
                        help='save root')

    args = parser.parse_args()
    train(args)
