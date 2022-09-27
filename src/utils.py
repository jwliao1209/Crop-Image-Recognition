import os
import glob
import time
import json
import torch
import random
import numpy as np

from datetime import datetime


def init_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


def get_time():
    time = datetime.today().strftime('%m-%d-%H-%M-%S')

    return time


def save_json(obj, path):
    with open(path, 'w') as fp:
        json.dump(vars(obj), fp, indent=4)

    return


def load_json(path):
    with open(path, 'r') as fp:
        obj = json.load(fp)

    return obj


def save_topk_ckpt(model, epoch, acc, save_dir, topk=5):
    save_name = f"ep={str(epoch):0>4}-acc={acc}.pth"
    model.module.save(os.path.join(save_dir, save_name))
    weight_list = sorted(
        glob.glob(os.path.join(save_dir, '*.pth')),
        key=lambda x: float(x[-10:-4]), reverse=True)

    # remove the last checkpoint except initial weight
    if len(weight_list) > topk+1:
        os.remove(weight_list[-2])

    return

