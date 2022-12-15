import os
import torch
import random
import numpy as np
import torch.nn.functional as F

from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms import (
    Resize,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    Normalize,
    AutoAugment, 
    RandomCrop,
    CenterCrop,
    Compose
    )
from monai.transforms import RandFlipd, ToTensord


class BaseTransform(object):
    def __init__(self, keys, **kwargs):
        self.keys = keys
        self._parse_var(**kwargs)

    def __call__(self, data, **kwargs):
        for key in self.keys:
            if key in data:
                data[key] = self._process(data[key], **kwargs)
            else:
                raise KeyError(f"{key} is not a key in data.")

        return data

    def _parse_var(self, **kwargs):
        pass

    def _process(self, single_data, **kwargs):
        NotImplementedError

    def _update_prob(self, cur_ep, total_ep):
        pass


class ReadImaged(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(ReadImaged, self).__init__(keys, **kwargs)

    def _process(self, single_data, **kwargs):
        single_data = read_image(single_data)
        return single_data


class ConcatMask():
    def __init__(self, **kwargs):
        super(ConcatMask, self).__init__()

    def __call__(self, data):
        data['image'] = torch.cat([
            data['image'], data['mask'][0, :, :].unsqueeze(0)], dim=0)

        return data


class MirrorPaddingd(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(MirrorPaddingd, self).__init__(keys, **kwargs)

    def _process(self, single_data, **kwargs):
        _, h, w = single_data.shape
        if h == w:  return single_data

        max_len = max(h, w)
        if h == max_len:  # h > w
            w_pad = max_len - w
            w_pad_half = w_pad // 2
            pad_size = (w_pad_half, w_pad-w_pad_half, 0, 0)
        else:  # w > h
            h_pad = max_len - h
            h_pad_half = h_pad // 2
            pad_size = (0, 0, h_pad_half, h_pad-h_pad_half)
        
        pad_image = F.pad(single_data.float(), pad_size, 'reflect')

        return pad_image.to(torch.uint8)


class RandomSquareCropd(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(RandomSquareCropd, self).__init__(keys, **kwargs)
    
    def _process(self, single_data):
        _, h, w = single_data.shape
        crop_size = min(h, w)
        croper = RandomCrop((crop_size, crop_size))

        return croper(single_data)


class SquareCentorCropd(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(SquareCentorCropd, self).__init__(keys, **kwargs)
    
    def _process(self, single_data):
        _, h, w = single_data.shape
        crop_size = min(h, w)
        croper = CenterCrop((crop_size, crop_size))

        return croper(single_data)


class CentorCropd(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(CentorCropd, self).__init__(keys, **kwargs)
        self.croper = CenterCrop(kwargs.get('size'))

    def _process(self, single_data):
        return self.croper(single_data)


class RandomResizedCropd(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(RandomResizedCropd, self).__init__(keys, **kwargs)

    def _parse_var(self, **kwargs):
        self.croper = RandomResizedCrop(kwargs.get('size'))

    def _process(self, single_data, **kwargs):
        return self.croper(single_data)


class ResizeImaged(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(ResizeImaged, self).__init__(keys, **kwargs)

    def _parse_var(self, **kwargs):
        self.resize = Resize(kwargs.get('size'))

    def _process(self, single_data, **kwargs):
        return self.resize(single_data)


class GridMask():
    def __init__(self, shape=(32, 32), dmin=5, dmax=10, ratio=0.7, p=0.3):
        self.shape = shape
        self.dmin = dmin
        self.dmax = dmax
        self.ratio = ratio
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img 
        d = random.randint(self.dmin, self.dmax)
        dx, dy = random.randint(0, d-1), random.randint(0, d-1)
        sl = int(d * (1-self.ratio))

        for i in range(dx, self.shape[0], d):
            for j in range(dy, self.shape[1], d):
                row_end = min(i+sl, self.shape[0])
                col_end = min(j+sl, self.shape[1])
                img[:, i:row_end, j:col_end] = 0

        return img

    def reset(self, h, w):
        self.shape = (h,w)
        self.dmin = np.min([h,w]) // 6
        self.dmax = np.max([h,w]) // 3

        return


class GridMaskd(BaseTransform):
    '''
    shape : the region might be drawn in black square masks, default is all pic
    dmin  : region of the black square mask (min), default min([h,w])//6
    dmax  : region of the black square mask (max), default min([h,w])//3
    ratio : the maintenance rate in the given square mask 
    p     : the probability of applying grid mask method
    '''
    def __init__(self, keys, **kwargs):
        super(GridMaskd, self).__init__(keys, **kwargs)
        self.grid_mask = GridMask()

    def _process(self, single_data, **kwargs):
        [c, h, w] = single_data.shape
        self.grid_mask.reset(h,w)

        return self.grid_mask(single_data)


class RandomFlipRot90(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(RandomFlipRot90, self).__init__(keys, **kwargs)

    def _parse_var(self, **kwargs):
        self.hflip = RandomHorizontalFlip(p=0.5)
        self.vflip = RandomVerticalFlip(p=0.5)
        self.rot90 = RandomRotation((90, 90))

    def _process(self, single_data, **kwargs):
        x = self.hflip(single_data)
        x = self.vflip(x)
        if random.random() < 0.5:
            x = self.rot90(x)

        return x


class AutoAugmentd(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(AutoAugmentd, self).__init__(keys, **kwargs)

    def _parse_var(self, **kwargs):
        self.p = kwargs.get('prob')
        self.autoaug = AutoAugment()

    def _process(self, single_data, **kwargs):
        if random.random() < self.p:
            return self.autoaug(single_data)
        else:
            return single_data


class NormalizeImaged(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(NormalizeImaged, self).__init__(keys, **kwargs)

    def _parse_var(self, **kwargs):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.normal = Normalize(mean, std)

    def _process(self, single_data, **kwargs):
        return self.normal(single_data.float())


class VisualizeImaged(BaseTransform):
    def __init__(self, keys, exit_opt=True, **kwargs):
        super(VisualizeImaged, self).__init__(keys, **kwargs)
        self.count = 0
        self.exit_opt = exit_opt
    
    def _process(self, single_data, **kwargs):
        os.makedirs('./tmp', exist_ok=True)
        save_image(single_data / 255, f'./cach/visual_image_{self.count}.jpg')
        self.count += 1
        print(single_data.shape)

        if self.exit_opt:
            exit()
        
        return single_data


def transforms_v1(args):
    train_transforms = Compose([
        ReadImaged(keys=['image']),
        ResizeImaged(keys=['image'],
                     size=(args.image_size, args.image_size)),
        RandFlipd(keys=['image'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image'], prob=0.5, spatial_axis=1),
        ToTensord(keys=['image', 'label']),
        AutoAugmentd(keys=['image'], prob=args.autoaug),
        NormalizeImaged(keys=['image']),
    ])

    val_transforms = Compose([
        ReadImaged(keys=['image']),
        ResizeImaged(keys=['image'],
                     size=(args.image_size, args.image_size)),
        ToTensord(keys=['image', 'label']),
        NormalizeImaged(keys=['image']),
        ])

    test_transforms = Compose([
        ReadImaged(keys=['image']),
        ResizeImaged(keys=['image'],
                     size=(args.image_size, args.image_size)),
        ToTensord(keys=['image']),
        NormalizeImaged(keys=['image']),
        ])

    return train_transforms, val_transforms, test_transforms


def transforms_v2(args):
    train_transforms = Compose([
        ReadImaged(keys=['image']),
        RandomResizedCropd(keys=['image'],
                           size=(args.image_size, args.image_size)),
        RandomFlipRot90(keys=['image']),
        ToTensord(keys=['image', 'label']),
        AutoAugmentd(keys=['image'], prob=args.autoaug),
        NormalizeImaged(keys=['image']),
    ])

    val_transforms = Compose([
        ReadImaged(keys=['image']),
        CentorCropd(keys=['image'], size=(args.image_size)),
        ToTensord(keys=['image', 'label']),
        NormalizeImaged(keys=['image']),
        ])

    test_transforms = Compose([
        ReadImaged(keys=['image']),
        CentorCropd(keys=['image'],
                    size=(args.image_size, args.image_size)),
        ToTensord(keys=['image']),
        NormalizeImaged(keys=['image']),
        ])

    return train_transforms, val_transforms, test_transforms


def get_transforms(args):
    transform_dict = dict(
        v1=transforms_v1,
        v2=transforms_v2
        )
    return transform_dict[args.trans](args)


if __name__ == '__main__':
    pass
