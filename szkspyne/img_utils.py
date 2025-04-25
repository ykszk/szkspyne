'''
Utility functions for image manipulation
'''

from typing import List, Sequence, Tuple, Union

import numpy as np
import skimage.transform
import torch
from szkmipy import boundingbox as bb


def pad_to_shape(x: np.ndarray, shape: Sequence[int], constant_value: float) -> np.ndarray:
    '''
    Add zero padding to bottom right.


    Examples:

    >>> a = np.ones((2, 2))
    >>> pad_to_shape(a, (3, 4), 0)
    array([[1., 1., 0., 0.],
           [1., 1., 0., 0.],
           [0., 0., 0., 0.]])
    '''
    ds = [ns - os for ns, os in zip(shape, x.shape)]
    return np.pad(x, [(0, d) for d in ds], constant_values=constant_value)


def pad_to_next_pow2(x: np.ndarray, axis: Sequence[int], constant_value: float) -> np.ndarray:
    '''
    Args:
        axis: list of axis

    Examples:

    >>> a = np.ones((2, 3))
    >>> pad_to_next_pow2(a, [1], 0)
    array([[1., 1., 1., 0.],
           [1., 1., 1., 0.]])
    '''
    new_shape = np.power(2, np.ceil(np.log2(x.shape))).astype(int)
    new_shape = [ns if i in axis else os for i, (ns, os) in enumerate(zip(new_shape, x.shape))]
    return pad_to_shape(x, new_shape, constant_value)


def pad_to_next_multiple(m: int, x: np.ndarray, axis: Sequence[int], constant_value: float) -> np.ndarray:
    '''
    shape a*m+b -> (a+1)*m

    Args:
        m: specified axis of returned shape will be multiple of this
        axis: list of axis

    Examples:

    >>> a = np.ones((1, 3))
    >>> pad_to_next_multiple(3, a, [0], 0)
    array([[1., 1., 1.],
           [0., 0., 0.],
           [0., 0., 0.]])
    '''
    new_shape = (m * np.ceil(np.array(x.shape) / m)).astype(np.int64)
    new_shape = [ns if i in axis else os for i, (ns, os) in enumerate(zip(new_shape, x.shape))]
    return pad_to_shape(x, new_shape, constant_value)


def pad_batch(xs: List[np.ndarray], factor: int, value):
    shapes = [x.shape for x in xs]
    max_shape = np.max(shapes, axis=0)
    new_shape = (factor * np.ceil(np.array(max_shape) / factor)).astype(np.int64).tolist()
    xs = [pad_to_shape(x, new_shape, value)[np.newaxis] for x in xs]
    return xs


def calc_resize_height(target_height, height, width):
    '''
    Calculate scale and width to resize image to the height. Width is dynamically determined (i.e. aspect ratio is kept.)
    '''
    scale = target_height / height
    width = np.round(scale * width).astype(int)
    return scale, width


def resize_image_height(height: int, img: np.ndarray) -> np.ndarray:
    _scale, width = calc_resize_height(height, img.shape[0], img.shape[1])
    img = skimage.transform.resize(img, (height, width), preserve_range=True)
    return img


def resize_height(
    height: int, img: np.ndarray, kps: Union[np.ndarray, List[np.ndarray], None] = None
) -> Union[Tuple[np.ndarray, List[np.ndarray]], np.ndarray]:
    '''
    Resize image to the height. Width is dynamically determined (i.e. aspect ratio is kept.)

    Args:
        height: target image height
        img: img

    Examples:

    >>> a = np.ones((512, 256), dtype=np.uint8)
    >>> resized = resize_height(768, a)
    >>> resized.shape
    (768, 384)
    >>> kps = np.array([[2, 2]])
    >>> resized, resized_kps = resize_height(768, a, kps)
    >>> resized_kps
    [array([3., 3.])]
    '''
    dtype = img.dtype
    assert dtype == np.uint8 or dtype == np.uint16, 'Invalid image dtype'
    scale, width = calc_resize_height(height, img.shape[0], img.shape[1])
    img = skimage.transform.resize(img, (height, width), preserve_range=True)
    img = np.round(img).astype(dtype)
    if kps is not None:
        kps = [scale * kp for kp in kps]
        return img, kps
    else:
        return img


class CropCollate:
    def __init__(self, margin=0, p=0.5, multiple=None, thresh: float = 1 / 1000):
        '''
        Crop images by the heatmap

        Args:
            multiple (integer):
        '''
        self.p = p
        self.margin = margin
        self.multiple = multiple
        self.thresh = thresh

    def __call__(self, batch):
        xs, ys = list(zip(*batch))
        xs, ys = np.array(xs), np.array(ys)
        if np.random.rand() < self.p:
            mask = ys > self.thresh
            bbox = bb.bbox(np.max(mask, axis=1, keepdims=True))
            xs = bb.crop(xs, bbox, self.margin)
            bbox = bb.bbox(mask)
            ys = bb.crop(ys, bbox, self.margin)
            if self.multiple:
                xs = pad_to_next_multiple(self.multiple, xs, [2, 3], 0)
                ys = pad_to_next_multiple(self.multiple, ys, [2, 3], 0)
        xs = torch.Tensor(xs)
        ys = torch.Tensor(ys)
        return xs, ys
