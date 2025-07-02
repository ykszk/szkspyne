import argparse
import random
import sys
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

from szkspyne import point_utils


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def percentile_normalize(x, percentile, clip=False, sampling_step=10):
    minmax = np.percentile(x.ravel()[::sampling_step], [percentile, 100 - percentile])
    x = (x - minmax[0]) / (minmax[1] - minmax[0])
    if clip:
        np.clip(x, 0, 1, out=x)
    return x


def base_image_transform(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        x = x / 255
    else:
        x = percentile_normalize(x, 1)
    if x.ndim == 2:
        x = x[np.newaxis]
    x = x.astype(np.float32)
    return x


def base_transform(x: np.ndarray, kps=None):
    x = base_image_transform(x)
    if kps is None:
        return x
    else:
        return x, kps


def base_kps_transform(img, kps, flip_pairs) -> Tuple[np.ndarray, point_utils.PointsList]:
    img, kps = base_transform(img, kps)
    return img, kps


def test_transform(img) -> np.ndarray:
    img = base_image_transform(img)
    return img


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
    def __init__(self, clipLimitChoices=[10, 100, 1000], gridSizeChoices=[8, 16, 32], p=0.5):
        super(RandomClahe, self).__init__(p=p)
        self.clipLimitChoices = clipLimitChoices
        self.gridSizeChoices = gridSizeChoices

    def apply(self, image, **params):
        clipLimit, gridSize = params['clipLimit'], params['gridSize']
        gridSize = (gridSize, gridSize)
        if image.dtype != np.uint8 or image.dtype != np.uint16:
            image = (image * 2048).astype(np.uint16)
        return (apply_clahe(image, clipLimit, gridSize) / 255).astype(np.float32)

    def get_params(self):
        return {'clipLimit': random.choice(self.clipLimitChoices), 'gridSize': random.choice(self.gridSizeChoices)}

    def get_transform_init_args_names(self):
        return ('clipLimitChoices', 'gridSizeChoices')


img_tfs = [
    RandomClahe(p=0.25),
    A.HorizontalFlip(p=0.25),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.25),
    A.Affine(scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)}, rotate=[-15, 15], shear=[-10, 10], p=0.25),
    A.GaussNoise(p=0.1),
    A.Blur(blur_limit=(3, 7), p=0.20),
    # A.Normalize(mean=0.485, std=0.229),
    ToTensorV2(),
]

# Use `A.ReplayCompose` to retrospectively access to the `applied` boolean parameter
album_transform = A.ReplayCompose(img_tfs, keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))

# CROP_HEIGHT, CROP_WIDTH = 512, 512
# CROPPER = A.RandomResizedCrop(height=CROP_HEIGHT, width=CROP_WIDTH, scale=[0.05, 0.20], ratio=[0.8, 1.2], p=1)
# album_crop_transform = A.ReplayCompose(
#     [CROPPER] + img_tfs, keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
# )

# album_base_crop_transform = A.ReplayCompose(
#     [A.CenterCrop(CROP_HEIGHT, CROP_WIDTH, always_apply=True), A.Normalize(mean=0.485, std=0.229)],
#     keypoint_params=A.KeypointParams(format='yx', remove_invisible=False),
# )


import warnings


def _random_transform(x: np.ndarray, keypoints, flip_pairs: List[Tuple[int, int]], tf_func, return_flipped=False):
    # print(1, x.shape)
    x, keypoints = base_transform(x, keypoints)
    x = x[0]  # remove channel dim
    # print(2, x.shape)
    # print(keypoints)
    sizes = [len(ps) for ps in keypoints]
    flat_points = point_utils.flatten_nest(keypoints)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tfed = tf_func(image=x, keypoints=flat_points)
    x, keypoints = tfed['image'], tfed['keypoints']
    keypoints = point_utils.partition(keypoints, sizes)
    flipped = False
    for tf in tfed['replay']['transforms']:
        if tf['__class_fullname__'] == 'HorizontalFlip':
            if tf['applied']:
                flipped = True
                for i, j in flip_pairs:
                    keypoints[i], keypoints[j] = keypoints[j], keypoints[i]
    # x = x[np.newaxis]
    # print('batch shape', x.shape)
    # print(3, x.shape)
    if return_flipped:
        return x, keypoints, flipped
    else:
        return x, keypoints


def random_transform(x: np.ndarray, keypoints, flip_pairs: List[Tuple[int, int]], return_flipped=False):
    return _random_transform(x, keypoints, flip_pairs, album_transform, return_flipped=return_flipped)


# def random_crop_transform(x: np.ndarray, keypoints, flip_pairs: List[Tuple[int, int]]):
#     return _random_transform(x, keypoints, flip_pairs, album_crop_transform)


# def base_crop_transform(x: np.ndarray, keypoints, flip_pairs: List[Tuple[int, int]]):
#     return _random_transform(x, keypoints, flip_pairs, album_base_crop_transform)
