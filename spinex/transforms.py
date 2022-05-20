import sys
import argparse
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def percentile_normalize(x, percentile, clip=False, sampling_step=10):
    minmax = np.percentile(x.ravel()[::sampling_step],
                           [percentile, 100 - percentile])
    x = (x - minmax[0]) / (minmax[1] - minmax[0])
    if clip:
        np.clip(x, 0, 1, out=x)
    return x


def base_transform(x, kps=None):
    if x.dtype == np.uint8:
        x = x / 255
    else:
        x = percentile_normalize(x, .1)
    if x.ndim == 2:
        x = x[np.newaxis]
    x = x.astype(np.float32)
    if kps is None:
        return x
    else:
        return x, kps


def invert_base(x):
    x = np.clip(x, 0, 1)
    x = np.round(x * 255, out=x)
    return x.astype(np.uint8)


def apply_clahe(img, clipLimit=100, tileGridSize=(32, 32)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(img)
    cl = cl - cl.min()
    cl = cl / cl.max()
    return np.round(np.clip(cl, 0, 1) * 255).astype(np.uint8)


class RandomClahe(ImageOnlyTransform):

    def __init__(self,
                 clipLimitChoices=[10, 100],
                 gridSizeChoices=[16, 32],
                 always_apply=False,
                 p=0.5):
        super(RandomClahe, self).__init__(always_apply, p)
        self.clipLimitChoices = clipLimitChoices
        self.gridSizeChoices = gridSizeChoices

    def apply(self, image, **params):
        clipLimit, gridSize = params['clipLimit'], params['gridSize']
        gridSize = (gridSize, gridSize)
        return apply_clahe(image, clipLimit, gridSize)

    def get_params(self):
        return {
            'clipLimit': random.choice(self.clipLimitChoices),
            'gridSize': random.choice(self.gridSizeChoices)
        }

    def get_transform_init_args_names(self):
        return ('clipLimitChoices', 'gridSizeChoices')


img_tfs = [
    # A.HorizontalFlip(p=0.25),
    RandomClahe(always_apply=True),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.2, p=.25),
    A.Affine(scale={
        'x': (.8, 1.2),
        'y': (.8, 1.2)
    },
             rotate=[-30, 30],
             shear=[-20, 20],
             p=.5),
    A.GaussNoise(var_limit=.02, p=.25),
    A.Normalize(mean=0.485, std=0.229),
    # ToTensorV2()
]
album_transform = A.Compose(img_tfs,
                            keypoint_params=A.KeypointParams(
                                format='yx', remove_invisible=False))


def random_transform(x, keypoints):
    # clipLimit = random.choice([10, 100])
    # gridSize = random.choice([16, 32])
    # x = apply_clahe(x, clipLimit, (gridSize, gridSize))
    # x, keypoints = base_transform(x, keypoints)
    # x = x.transpose((1, 2, 0))  # to channel last
    tfed = album_transform(image=x, keypoints=keypoints)
    x, keypoints = tfed['image'], tfed['keypoints']
    # x = x.transpose((2, 0, 1))  # to channel first
    x = x[np.newaxis]
    # return x.astype(np.float32), keypoints
    return x, keypoints
