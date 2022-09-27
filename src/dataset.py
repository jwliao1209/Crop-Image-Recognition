import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import (Compose, AddChanneld, ToTensord, RandRotate90, RandFlipd,
                              RandCropByPosNegLabeld, CropForegroundd, EnsureTyped, RandSpatialCropSamplesd)
from monai import transforms
from src.transforms import *

# data_file : train, valid 再切換抓不同模式的時候才能分辨
# opt.type  : train, valid mode

Lab2Cat = {1: 'asparagus',     2: 'bambooshoots',     3: 'betel',
 4: 'broccoli',      5: 'cauliflower',      6: 'chinesecabbage',
 7: 'chinesechives', 8: 'custardapple',     9: 'grape',
 10: 'greenhouse',  11: 'greenonion',      12: 'kale',
 13: 'lemon',       14: 'lettuce',         15: 'litchi',
 16: 'longan',      17: 'loofah',          18: 'mango',
 19: 'onion',       20: 'others',          21: 'papaya',
 22: 'passionfruit',23: 'pear',            24: 'pennisetum',
 25: 'redbeans',    26: 'roseapple',       27: 'sesbania',
 28: 'soybeans',    29: 'sunhemp',         30: 'sweetpotato',
 31: 'taro',        32: 'tea',             33: 'waterbamboo'}

Cat2Lab = dict([(Lab2Cat[key], key) for key in Lab2Cat])

class CropDataset(Dataset):
    def __init__(self, data_file, opt, transform=None):
        super(CropDataset, self).__init__()
        self.opt = opt
        self.data_file = data_file
        self.filename = list(data_file.filename)
        self.folder = list(data_file.folder)
        self.transform = transform

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, i):
        # omt_file = random.choice(self.data_file)
        filename = self.filename[i]
        folder = self.folder[i]
        data = {
            'filename': filename,
            'image': os.path.join(self.opt.root, folder, filename),
            'label': Cat2Lab[folder]-1,
            'target': []
        }

        if self.transform is not None:
            data = self.transform(data)
        return data


def get_train_val_dataset(opt):
    df = pd.read_csv(os.path.join('index', f"fold_{opt.folder}.csv"))
    trainframe = df[df["Type"]=="train"]
    validframe = df[df["Type"]=="valid"]

    train_transforms = Compose([
        read_imaged(keys=["image"]),
        Resized(keys=['image']),
        transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        transforms.ToTensord(keys=["image", "label"]),
        RandGaussNoised(keys=['image'], p=opt.prob_noise)
    ])

    val_transforms = Compose([
        read_imaged(keys=["image"]),
        Resized(keys=['image']),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.ToTensord(keys=["image", "label"])
        ])

    train_set = CropDataset(trainframe, opt, train_transforms)
    val_set = CropDataset(validframe, opt, val_transforms)
    return train_set, val_set





