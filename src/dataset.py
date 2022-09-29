import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, ToTensor, Normalize, RandomResizedCrop, CenterCrop

# from src.transforms import RandomNoise, GridMask, AutoAugmentation

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from monai.transforms import (Compose, AddChanneld, ToTensord, RandRotate90, RandFlipd,
                              RandCropByPosNegLabeld, CropForegroundd, EnsureTyped, RandSpatialCropSamplesd)
from monai import transforms
from src.transforms import *


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
    trainframe = df[df["Type"]=="train"].iloc[:opt.train_num]
    validframe = df[df["Type"]=="valid"].iloc[:opt.valid_num]

    train_transforms = Compose([
        read_imaged(keys=["image"]),
        Resized(keys=['image'], size=(opt.img_size, opt.img_size)),
        #transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        #transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        #transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        #transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        transforms.ToTensord(keys=["image", "label"]),
        #RandGaussNoised(keys=['image'], p=opt.prob_noise)
    ])

    val_transforms = Compose([
        read_imaged(keys=["image"]),
        Resized(keys=['image'], size=(opt.img_size, opt.img_size)),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.ToTensord(keys=["image", "label"])
        ])

    train_set = CropDataset(trainframe, opt, train_transforms)
    val_set = CropDataset(validframe, opt, val_transforms)
    return train_set, val_set

def get_train_val_loader(opt):
    train_set, val_set = get_train_val_dataset(opt)
    train_loader = DataLoader(train_set,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers)
    val_loader = DataLoader(val_set,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers)
    return train_loader, val_loader

    
'''
def get_train_val_loader(args):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    train_transform = Compose([
        RandomResizedCrop(args.img_size),
        RandomHorizontalFlip(p=args.fliplr),
        RandomRotation(degrees=args.rot_degree),
        AutoAugmentation(opt=args.autoaugment),
        ToTensor(),
        Normalize(mean, std),
        GridMask(),
        RandomNoise(p=args.noise)
        
    ])

    val_transform = Compose([
        CenterCrop(args.img_size),
        ToTensor(),
        Normalize(mean, std),
    ])

    data_dir = './datasets'
    train_set = datasets.ImageFolder(os.path.join(data_dir, f'fold{args.fold}', 'train'), transform=train_transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, f'fold{args.fold}', 'val'), transform=val_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def get_test_loader(args):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_transform = Compose([
        CenterCrop(480), # (args.img_size)
        ToTensor(),
        Normalize(mean, std),
    ])

    data_dir = './datasets'
    test_set = datasets.ImageFolder(os.path.join(data_dir, f'fold{args.fold}', 'test'), transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)
    img_list = [os.path.basename(list(test_set.imgs[i])[0]) for i in range(len(test_set))]

    return test_loader, img_list
'''    

