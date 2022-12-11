import os
import numpy as np
from math import pi
import torch
import torch.nn.functional as F
from monai.transforms import (Transform,
                              MapTransform, 
                              Compose)
from torchvision.io import read_image
from torchvision.transforms import Resize, Normalize, AutoAugment, RandomRotation

class BaseTransform(object):
    def __init__(self, keys, **kwargs):
        self.keys = keys
        self._parseVariables(**kwargs)

    def __call__(self, data, **kwargs):
        for key in self.keys:
            if key in data:
                data[key] = self._process(data[key], **kwargs)
            else:
                raise KeyError(f"{key} is not a key in data.")
        
        return data
    
    def _parseVariables(self, **kwargs):
        pass

    def _process(self, single_data, **kwargs):
        pass

    def _update_prob(self, cur_ep, total_ep):
        pass

class MirrorPaddingd(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(MirrorPaddingd, self).__init__(keys, **kwargs)

    def _process(self, single_data, **kwargs):
        _, h, w = single_data.shape
        if h == w: return single_data

        max_len = max(h, w)
        if h == max_len: # h > w
            w_pad = max_len - w
            w_pad_half = w_pad // 2
            pad_size = (w_pad_half, w_pad-w_pad_half, 0, 0)
        else:
            h_pad = max_len - h
            h_pad_half = h_pad // 2
            pad_size = (0, 0, h_pad_half, h_pad-h_pad_half)

        pad_image = F.pad(single_data.float(), pad_size, 'reflect')
        return pad_image.to(torch.uint8)



class Load(MapTransform):
    def __init__(self, keys):
        self.keys = keys
    def __call__(self, data):
        d = dict(data)
        d["image"] = read_image(d[self.keys[0]])
        return d

class Scale01d(MapTransform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        d["image"] = d["image"].float()/255
        return d

class Normalized(MapTransform):
    def __init__(self, keys):
        self.keys = keys
        # self.normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.normalize = Normalize((0.457, 0.486, 0.423), (0.253, 0.249, 0.284))

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key] = self.normalize(d[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return d

class Resized(MapTransform):
    def __init__(self, keys, size=(384, 384)):
        self.keys = keys
        self.resize = Resize(size)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key] = self.resize(d[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return d

class AutoAugmentd(MapTransform):
    def __init__(self, keys):
        self.keys = keys
        self.autoaug = AutoAugment()
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key] = self.autoaug(d[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return d 

class RandomRotationd(MapTransform):
    def __init__(self, keys, degrees=180):
        self.keys = keys
        self.rotate = RandomRotation(degrees)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key] = self.rotate(d[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')
        return d
    
def train_transforms(**transforms_config):
    img_size = transforms_config.get("img_size", 224)
    transforms = [
        Load(keys=["image"]),
        # MirrorPaddingd(keys=["image"]),
        Resized(keys=["image"], size=(img_size, img_size))
    ]
    if transforms_config.get("random_rotation", False) == True:
        transforms += [RandomRotationd(keys=["image"], degrees=180)]

    if transforms_config.get("random_transform", False) == True:
        transforms += [AutoAugmentd(keys=["image"])]

    transforms += [
        Scale01d(keys=["image"]),
        Normalized(keys=["image"]),
    ]
    return Compose(transforms)

def val_transforms(**transforms_config):
    img_size = transforms_config.get("img_size", 224)
    transforms = [
        Load(keys=["image"]),
        Resized(keys=["image"], size=(img_size, img_size)),
        Scale01d(keys=["image"]),
        Normalized(keys=["image"]),
    ]
    
    return Compose(transforms)
