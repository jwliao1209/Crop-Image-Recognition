import torch
import random
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import Resize, Normalize, AutoAugment

__all__ = ["ReadImaged", "ResizeImaged", "AutoAugmentd", "NormalizeImaged"]


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
        NotImplementedError

    def _update_prob(self, cur_ep, total_ep):
        pass


class ReadImaged(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(ReadImaged, self).__init__(keys, **kwargs)

    def _process(self, single_data, **kwargs):
        return read_image(single_data)


class ResizeImaged(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(ResizeImaged, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
        self.resize = Resize(kwargs.get('size'))

    def _process(self, single_data, **kwargs):
        return self.resize(single_data)


class AutoAugmentd(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(AutoAugmentd, self).__init__(keys, **kwargs)

    def _parseVariables(self, **kwargs):
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

    def _parseVariables(self, **kwargs):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.normal = Normalize(mean, std)

    def _process(self, single_data, **kwargs):
        return self.normal(single_data.float())


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
        self.dmin = np.min([h,w])//6
        self.dmax = np.max([h,w])//3


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



##########################################################################################
from torchvision.transforms import (RandomHorizontalFlip,
                                    RandomOrder,
                                    RandomRotation,
                                    RandomPerspective,
                                    RandomResizedCrop,
                                    RandomAffine,
                                    RandomCrop,
                                    )


class RandomNoise():
    ''' Random noise from Gaussian distribution'''
    def __init__(self, sig=0.005, p=0.1):
        self.sig = sig
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            image += self.sig * torch.randn(image.shape)

        return image


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








class RandHoriFlipd():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = RandomHorizontalFlip()(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return data


class RandRotd():
    def __init__(self, keys, degrees=10):
        self.keys = keys
        self.degrees = degrees

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = RandomRotation(self.degrees)(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return data




class RandAffined():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = RandomAffine(0, shear=10,
                                         scale=(0.8, 1.2))(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return data


class CenterCropd():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = CenterCrop(200)(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return data


class RandomCropd():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = RandomCrop((375, 375))(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return data


class Normalized():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = Normalize((0.485, 0.456, 0.406),
                                      (0.229, 0.224, 0.225))(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return data


class Scale01d():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = data[key].float()/255
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return data


class RandomGaussianNoise():
    def __init__(self, sig=0.01, p=0.5):
        self.sig = sig
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            img += self.sig * torch.randn(img.shape)

        return img


class RandGaussNoised(RandomGaussianNoise):
    def __init__(self, keys , p):
        super().__init__()
        self.keys = keys
        self.p = p

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = RandomGaussianNoise(self.p)(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return data


class GridMask():
    def __init__(self, dmin=90, dmax=160, ratio=0.8, p=0.6):
        """Original Setting : dmin=90, dmax=300, ratio=0.7, p=0.5
        after augmentation, again masking with (90, 160, 0.8, 0.6)
        [ dmin, dmax ] : range of the d in uniform random
        random variable probibilaty > p, swith on the function"""
        self.dmin = dmin
        self.dmax = dmax
        self.ratio = ratio
        self.p = p

    def __call__(self, Img):
        if random.random() < self.p:
            return Img
        d = random.randint(self.dmin, self.dmax)
        dx, dy = random.randint(0, d-1), random.randint(0, d-1)
        sl = int((1-self.ratio)*d)
        for i in range(dx, Img.shape[1], d):
            for j in range(dy, Img.shape[2], d):
                row_end = min(i+sl, Img.shape[1])
                col_end = min(j+sl, Img.shape[2])
                Img[:, i:row_end, j:col_end] = 0

        return Img


class GridMaskd(GridMask):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = GridMask()(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')
        return data



import random
from torchvision import transforms as T
from PIL import ImageFilter, ImageOps

INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)
ORCHID_MEAN = (0.4909, 0.4216, 0.3703)
ORCHID_STD = (0.2459, 0.2420, 0.2489)


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p
        self.transf = T.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img

if __name__ == '__main__':
    pass
