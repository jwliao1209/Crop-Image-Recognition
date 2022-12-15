import os
import copy
import torch
import numpy as np
import pandas as pd

from glob import glob
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

from .transforms import get_transforms
from .utils import load_json
from .constant import DATA_ROOT, TEST_BS


DEBUG_TRAIN_NUM = 1000
DEBUG_VAL_NUM   = 1000


class CropDataset(Dataset):
    def __init__(self, data_list, transform=None):
        super(CropDataset, self).__init__()
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        data = copy.deepcopy(self.data_list[i])
        return self.transform(data) if self.transform is not None else data


def get_train_val_loader(args):
    data_list = load_json(os.path.join(DATA_ROOT, f"fold_{args.fold}.json"))
    train_transforms, val_transforms, _ = get_transforms(args)
    train_list = data_list['train'][:args.train_num] if args.train_num > 0 else data_list['train']
    val_list = data_list['val'][:args.val_num] if args.val_num > 0 else data_list['val']
    args.train_num = len(train_list)
    train_set = CropDataset(train_list, train_transforms)
    val_set = CropDataset(val_list, val_transforms)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
        )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
        )

    return train_loader, val_loader


def get_test_loader(args, test_type='public'):
    data_list = load_json(os.path.join(DATA_ROOT, f"{test_type}.json"))
    _, _, test_transforms = get_transforms(args)
    test_set = CropDataset(data_list[test_type], test_transforms)
    test_loader = DataLoader(test_set, batch_size=TEST_BS, shuffle=False, num_workers=8)
    print(f'Number of {test_type} data: {len(test_set)}')

    return test_loader
