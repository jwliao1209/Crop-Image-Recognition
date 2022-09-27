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


'''
import torch
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


class RandomNoise():
    # Random noise from Gaussian distribution
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


class AutoAugmentation():
    def __init__(self, fillcolor=(128, 128, 128), opt=0):
        self.opt = opt
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        if self.opt == 0:
            return img

        else:
            policy_idx = random.randint(0, len(self.policies)-1)

            return self.policies[policy_idx](img)
        
    def __repr__(self):
        return "AutoAugmentation"


class SubPolicy():
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        func = {
            "shearX": ShearX(fillcolor=fillcolor),
            "shearY": ShearY(fillcolor=fillcolor),
            "translateX": TranslateX(fillcolor=fillcolor),
            "translateY": TranslateY(fillcolor=fillcolor),
            "rotate": Rotate(),
            "color": Color(),
            "posterize": Posterize(),
            "solarize": Solarize(),
            "contrast": Contrast(),
            "sharpness": Sharpness(),
            "brightness": Brightness(),
            "autocontrast": AutoContrast(),
            "equalize": Equalize(),
            "invert": Invert()
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class ShearX():
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)


class ShearY():
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)


class TranslateX():
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=self.fillcolor)


class TranslateY():
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * x.size[1] * random.choice([-1, 1])),
            fillcolor=self.fillcolor)


class Rotate():
    def __call__(self, x, magnitude):
        rot = x.convert("RGBA").rotate(magnitude * random.choice([-1, 1]))
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(x.mode)


class Color():
    def __call__(self, x, magnitude):
        return ImageEnhance.Color(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Posterize():
    def __call__(self, x, magnitude):
        return ImageOps.posterize(x, magnitude)


class Solarize():
    def __call__(self, x, magnitude):
        return ImageOps.solarize(x, magnitude)


class Contrast():
    def __call__(self, x, magnitude):
        return ImageEnhance.Contrast(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Sharpness():
    def __call__(self, x, magnitude):
        return ImageEnhance.Sharpness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Brightness():
    def __call__(self, x, magnitude):
        return ImageEnhance.Brightness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class AutoContrast():
    def __call__(self, x, magnitude):
        return ImageOps.autocontrast(x)


class Equalize():
    def __call__(self, x, magnitude):
        return ImageOps.equalize(x)


class Invert():
    def __call__(self, x, magnitude):
        return ImageOps.invert(x)


if __name__ == '__main__':
    pass
'''
