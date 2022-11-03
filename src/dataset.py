import os
import torch
import numpy as np
import pandas as pd

from glob import glob
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from monai.transforms import RandFlipd, ToTensord

from .transforms import ReadImaged, ResizeImaged, AutoAugmentd, NormalizeImaged
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
        data = self.data_list[i]        
        return self.transform(data) if self.transform else data


def get_train_val_loader(args):
    data_list = load_json(os.path.join(DATA_ROOT, f"fold_{args.fold}.json"))
    train_transforms = get_train_transforms_v1(args)
    val_transforms = get_val_transforms_v1(args)
    train_set = CropDataset(data_list['train'], train_transforms)
    val_set = CropDataset(data_list['val'][:DEBUG_VAL_NUM], val_transforms)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def get_test_loader(args, test_type='public'):
    data_list = load_json(os.path.join(DATA_ROOT, f"{test_type}.json"))
    test_transforms = get_test_transforms_v1(args)
    test_set = CropDataset(data_list[test_type], test_transforms)    
    test_loader = DataLoader(test_set, batch_size=TEST_BS, shuffle=False, num_workers=8)

    return test_loader


def get_train_transforms_v1(args):
    train_transforms = Compose([
        ReadImaged(keys=['image']),
        ResizeImaged(keys=['image'],
                     size=(args.image_size, args.image_size)),
        RandFlipd(keys=['image'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image'], prob=0.5, spatial_axis=1),
        AutoAugmentd(keys=['image'], prob=args.autoaug),
        ToTensord(keys=['image', 'label']),
        NormalizeImaged(keys=['image']),
    ])

    return train_transforms


def get_val_transforms_v1(args):
    val_transforms = Compose([
        ReadImaged(keys=['image']),
        ResizeImaged(keys=['image'],
                     size=(args.image_size, args.image_size)),
        ToTensord(keys=['image', 'label']),
        NormalizeImaged(keys=['image']),
        ])

    return val_transforms


def get_test_transforms_v1(args):
    test_transforms = Compose([
        ReadImaged(keys=['image']),
        ResizeImaged(keys=['image'],
                     size=(args.image_size, args.image_size)),
        ToTensord(keys=['image']),
        NormalizeImaged(keys=['image']),
        ])

    return test_transforms
