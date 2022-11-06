import os
import re
import csv
import glob
import time
import json
import torch
import random
import argparse
import numpy as np
from datetime import datetime


WEIGHT_FORMAT = 'ep={0:0>4}-acc={1:.4f}.pth'
ACC_REGEX = r"(?<=acc=)\d+\.?\d*"


def debug_fun(*args, **kwargs):
    for e in args:
        print(e)

    for k,v in kwargs.item():
        print(k, v)

    exit()

    return


def fixed_random_seed(seed):
    def decorator(func):
        def wrap(*args, **kargs):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            func(*args, **kargs)
        return wrap

    return decorator


def generate_dir(func):
    def wrap(path, *args):
        dir_ = os.path.dirname(path)
        os.makedirs(dir_, exist_ok=True)
        func(path, *args)
    return wrap


def get_time():
    return datetime.today().strftime('%m-%d-%H-%M-%S')


def init_models(models, ckpts, device):
    for m, c in zip(models, ckpts):
        m.load(c)
        m.to(device)
        m.eval()

    return models


@generate_dir
def save_json(path, obj):
    with open(path, 'w') as fp:
        json.dump(vars(obj), fp, indent=4)

    return


def load_json(path):
    with open(path, 'r') as fp:
        obj = json.load(fp)

    return obj


def save_topk_ckpt(model, epoch, acc, save_dir, topk=5):
    os.makedirs(save_dir, exist_ok=True)
    save_name = WEIGHT_FORMAT.format(epoch, acc)
    model.module.save(os.path.join(save_dir, save_name))
    weight_list = sorted(
        glob.glob(os.path.join(save_dir, '*.pth')),
        key=lambda x: float(
            re.findall(ACC_REGEX, x)[0]
            ), reverse=True)

    # remove the last checkpoint except initial weight
    if len(weight_list) > topk+1:
        os.remove(weight_list[-2])

    return


def get_topk_ckpt(weight_path, topk):
    topk_ckpt = []
    for ckpt, tk in zip(weight_path, topk):
        weight_list = sorted(
            glob.glob(os.path.join('checkpoint', ckpt, 'weight', '*.pth')),
            key=lambda x: float(
                re.findall(ACC_REGEX, x)[0]
                ), reverse=True)
        topk_ckpt += weight_list[:tk]

    return topk_ckpt


def get_ckpt_config_args(args):
    config = load_json(
        os.path.join('checkpoint', args.checkpoint[0], 'config.json'))
    args = argparse.Namespace(**vars(args), **config)

    return args


@generate_dir
def save_csv(save_path, data_dict_list):
    fieldnames = data_dict_list[0].keys()
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_dict_list)
    
    return


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.count = 0
        self.total = 0
        self.avg = 0
        self.avg_str = 0

        return

    def update(self, value, batch_size=1):
        self.value = value
        self.count += batch_size
        self.total += value * batch_size
        self.avg = self.total / self.count
        self.avg_str = f"{self.avg:.4f}"

        return


class Recorder():
    def __init__(self, ep, mode):
        self.cur_ep = ep
        self.mode = mode
        self.loss = AverageMeter()
        self.acc = AverageMeter()
        self.lr = 0
    
    def update(self, loss, acc, bs, lr):
        self.loss.update(loss, bs)
        self.acc.update(acc, bs)
        self.lr = lr

        return
    
    def get_iter_record(self):
        record = {}
        record['loss'] = self.loss.avg_str
        record['acc'] = self.acc.avg_str
        record['lr'] = f'{self.lr:.8f}'

        return record
    
    def get_epoch_record(self):
        return {'epoch': self.cur_ep, 'type': self.mode, **self.get_iter_record()}
