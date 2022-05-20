'''
Utility functions for point manipulation
'''

from typing import Sequence, Callable, Tuple, Optional, TypeVar

import numpy as np
import scipy
import scipy.stats
import torch
import torch.utils.data
from numba import njit
import itertools

PointType = np.ndarray  # Tuple[float, float]
PointsType = Sequence[PointType]
PointsList = Sequence[PointsType]
ImageType = np.ndarray

GenericType = TypeVar('GenericType')


def flatten_nest(l: Sequence[Sequence[GenericType]]) -> Sequence[GenericType]:
    '''
    Flatten nested list

    Examples:

    >>> a = [[1, 2], [3, 4, 5]]
    >>> flatten_nest(a)
    [1, 2, 3, 4, 5]
    '''
    return list(itertools.chain.from_iterable(l))


def partition(lst: Sequence[GenericType],
              sizes: Sequence[int]) -> Sequence[Sequence[GenericType]]:
    '''
    Partition list into sub lists with given sizes

    Examples:

    >>> a = [1, 2, 3, 4, 5]
    >>> partition(a, [2, 3])
    [[1, 2], [3, 4, 5]]
    '''
    cs = np.cumsum(sizes)
    cs = np.insert(cs, 0, 0)  # prepend 0
    assert len(lst) == cs[-1]
    return [lst[cs[i]:cs[i + 1]] for i in range(len(cs) - 1)]

def flat_apply_partition(lst: Sequence[Sequence[GenericType]], f) -> Sequence[Sequence[GenericType]]:
    sizes = [len(e) for e in lst]
    flat = flatten_nest(lst)
    result = f(flat)
    return partition(result, sizes)


class ImageDataset(torch.utils.data.Dataset):
    '''
        Dataset for image without points
    '''

    def __init__(self, xs: Sequence[np.ndarray],
                 transform: Optional[Callable[[np.ndarray], np.ndarray]]):
        '''
        Args:
            xs: list of x.
            transform: transformation.
        '''
        self.xs = xs
        self.transform = transform

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = self.xs[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x


class KeypointDataset(torch.utils.data.Dataset):
    '''
        Dataset for image and keypoints
    '''

    def __init__(self, xs: Sequence[np.ndarray], points: Sequence[PointsList],
                 point_pairs,
                 transform: Optional[Callable[[np.ndarray, PointsType],
                                              Tuple[np.ndarray, PointsType]]],
                 sigma: float):
        '''
        Args:
            xs: list of x.
            points: list of 2D points.
            point_pairs: pair of 2D points.
            transform: transformation.
            sigma: sigma for heatmap generation
        '''
        self.xs = xs
        self.points = points
        if point_pairs is None:
            self.point_pairs = [[]] * len(xs)
        else:
            self.point_pairs = point_pairs
        self.transform = transform
        self.sigma = sigma

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x, points, point_pairs = self.xs[idx], self.points[
            idx], self.point_pairs[idx]
        if self.transform is not None:
            sizes = [len(ps) for ps in points]
            flat_points = flatten_nest(points)
            x, points = self.transform(x, flat_points)
            points = partition(points, sizes)

        n_chs = len(points) + len(point_pairs)
        y = np.zeros((n_chs, x.shape[1], x.shape[2]), dtype=np.float32)
        for i, p in enumerate(points):
            create_prob_image(x.shape[1:], p, self.sigma, out=y[i])
        # TODO: make affinity map using self.point_pairs
        for i, pairs in enumerate(point_pairs, len(points)):
            raise NotImplementedError

        return x, y


def multivariate_normal(shape: Sequence[int], mu: np.ndarray,
                        sigma: float) -> np.ndarray:
    '''
    Generate PDF(heatmap) for multivariate gaussian
    
    Args:
        shape (array) : output shape
        mu (array) : mean/center
        sigma (array or scalar): standard deviation
    '''
    if np.isscalar(mu):
        mu = np.repeat(mu, len(shape))
    vecs = [
        scipy.stats.norm.pdf(np.arange(size), m, sigma)
        for size, m in zip(shape, mu)
    ]
    # prepare for broadcasting
    vecs = [
        v.reshape([1 if j != i else s for j in range(len(vecs))])
        for i, (v, s) in enumerate(zip(vecs, shape))
    ]
    if len(shape) == 1:
        return vecs[0]

    output = vecs[0]
    for v in vecs[1:]:
        output = output * v
    return output


def create_prob_image(shape: Sequence[int],
                      points: Sequence[PointType],
                      sigma: float,
                      normalize: bool = True,
                      out: Optional[np.ndarray] = None) -> np.ndarray:
    '''
    Create heatmap from the given points
    '''
    if out is None:
        out = multivariate_normal(shape, points[0], sigma)
    else:
        out[...] = multivariate_normal(shape, points[0], sigma)
    for p in points[1:]:
        np.maximum(out, multivariate_normal(shape, p, sigma), out=out)
    if normalize and out.max() != 0:
        out *= (1 / out.max())
    assert out is not None  # type narrowing
    return out


@njit("(i8, i8, f8[:], f8[:], f8, f4[:,:])", cache=True, fastmath=True)
def _gauss_line_dist(w, h, a, b, sigma, img):
    a_y, a_x, = a
    ab = b - a
    ab_y, ab_x, = ab
    l = (ab**2).sum()
    coef = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    denom = 1.0 / (2 * sigma * sigma)

    for y in range(h):
        img_row = img[y]
        pa_y = y - a_y
        for x in range(w):
            pa_x = x - a_x
            dot = pa_x * ab_x + pa_y * ab_y
            t = min(max(0.0, dot / l), 1.0)  # clip [0, 1]
            proj_x = a_x + t * ab_x
            proj_y = a_y + t * ab_y
            d_x = x - proj_x
            d_y = y - proj_y
            dist_sq = d_x * d_x + d_y * d_y
            g = coef * np.exp(-dist_sq * denom)
            img_row[x] = g


def gauss_line_dist(shape: Sequence[int],
                    p1: PointType,
                    p2: PointType,
                    sigma: float,
                    normalize: bool = True,
                    out: Optional[np.ndarray] = None) -> np.ndarray:
    '''
    Craete affinity map between two points
    '''
    if out is None:
        out = np.zeros(shape, dtype=np.float32)
    _gauss_line_dist(shape[1], shape[0], np.array(p1, dtype=np.float64),
                     np.array(p2, dtype=np.float64), sigma, out)
    if normalize and out.max() != 0:
        out *= (1 / out.max())
    assert out is not None  # type anrrowing
    return out
