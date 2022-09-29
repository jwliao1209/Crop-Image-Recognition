import torch
import random
from torchvision.io import read_image
from torchvision.transforms import (Resize,
                                    RandomOrder,
                                    RandomHorizontalFlip,
                                    RandomRotation,
                                    RandomPerspective,
                                    RandomResizedCrop,
                                    RandomAffine,
                                    RandomCrop,
                                    Normalize,
                                    Compose)
'''
Simulate monai's Dictionary Transforms writing
https://docs.monai.io/en/latest/transforms.html#dictionary-transforms
'''


class read_imaged():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = read_image(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return data


class Resized():
    def __init__(self, keys, size=(384, 384)):
        self.keys = keys
        self.size = size

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = Resize(self.size)(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')

        return data


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


class Transforms():
    def __init__(self, size=500):
        self.size = size

    def train_transforms(self):
        train_trans = Compose([read_imaged(keys=['image']),
                               Scale01d(keys=['image']),
                               Resized(keys=['image'],
                                       size=(self.size, self.size)),
                               RandGaussNoised(keys=['image']),
                               RandomOrder(
                                   [RandHoriFlipd(keys=['image']),
                                    RandRotd(keys=['image'], degrees=10),
                                    RandAffined(keys=['image'])]),
                               GridMaskd(keys=['image']),
                               Normalized(keys=['image'])])

        return train_trans

    def valid_transforms(self):
        valid_trans = Compose([read_imaged(keys=['image']),
                               Scale01d(keys=['image']),
                               Resized(keys=['image'],
                                       size=(self.size, self.size)),
                               Normalized(keys=['image'])])

        return valid_trans

    def test_transforms(self):
        test_trans = Compose([read_imaged(keys=['image']),
                              Scale01d(keys=['image']),
                              Resized(keys=['image'],
                                      size=(self.size, self.size)),
                              Normalized(keys=['image'])])

        return test_trans

# https://github.com/Rammstein-1994/orchid_competition/blob/main/src/transform.py
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


def train_tf(
    img_size: int = 224, three_data_aug: bool = False, color_jitter: float = None
) -> T.Compose:

    primary_tf1 = [
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomCrop(img_size, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
    ]

    if three_data_aug:
        primary_tf1 += [
            T.RandomChoice(
                [gray_scale(p=0.5), Solarization(p=0.5), GaussianBlur(p=0.5)]
            )
        ]

    if color_jitter:
        primary_tf1.append(T.ColorJitter(color_jitter, color_jitter, color_jitter))

    final_tfl = [T.ToTensor(), T.Normalize(mean=ORCHID_MEAN, std=ORCHID_STD)]

    return T.Compose(primary_tf1 + final_tfl)


def test_tf(img_size: int = 224) -> T.Compose:
    size = int((256 / 224) * img_size)
    return T.Compose(
        [
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=ORCHID_MEAN, std=ORCHID_STD),
        ]
    )

