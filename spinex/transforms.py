import sys
import argparse
import random

import albumentations as A
import numpy as np
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


# https://albumentations.ai/docs/examples/example_keypoints/
def vis_keypoints(image, keypoints, color=(0, 255, 0), diameter=15):
    image = image.copy()

    for (y, x) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)

    return image


class CoarseDropout(A.CoarseDropout):
    '''
    Apply identity transform to keypoints
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint


img_tfs = [
    # CoarseDropout(
    #     max_holes=50, max_height=64, max_width=64, fill_value=.5, p=.2),
    A.HorizontalFlip(p=0.25),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.2, p=.5),
    A.Affine(scale={
        'x': (.8, 1.2),
        'y': (.8, 1.2)
    },
             rotate=[-30, 30],
             shear=[-20, 20],
             p=.8),
    A.GaussNoise(var_limit=.02, p=.5)
]
album_transform = A.Compose(img_tfs,
                            keypoint_params=A.KeypointParams(
                                format='yx', remove_invisible=False))

album_transform_wo_kps = A.Compose(img_tfs)


def percentile_normalize(x, percentile, clip=False):
    minmax = np.percentile(
        x.ravel()[::10],
        [percentile, 100 - percentile])  # ::10 to speedup computation
    x = (x - minmax[0]) / (minmax[1] - minmax[0])
    if clip:
        np.clip(x, 0, 1, out=x)
    return x


def base_transform(x, kps=None):
    if x.dtype == np.uint8:
        x = x / 255
    else:
        x = percentile_normalize(x, .001)
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


def random_transform(x, keypoints):
    clipLimit = random.choice([10, 100])
    gridSize = random.choice([16, 32])
    x = apply_clahe(x, clipLimit, (gridSize, gridSize))
    x, keypoints = base_transform(x, keypoints)
    x = x.transpose((1, 2, 0))  # to channel last
    tfed = album_transform(image=x, keypoints=keypoints)
    x, keypoints = tfed['image'], tfed['keypoints']
    x = x.transpose((2, 0, 1))  # to channel first
    return x.astype(np.float32), keypoints


def apply_clahe(img, clipLimit=100, tileGridSize=(32, 32)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(img)
    cl = cl - cl.min()
    cl = cl / cl.max()
    return np.round(np.clip(cl, 0, 1) * 255).astype(np.uint8)
