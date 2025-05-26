'''
Utility functions for point manipulation
'''

import itertools
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import scipy
import scipy.stats
import torch
import torch.utils.data
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.measurements import center_of_mass
from skimage import measure
from szkmipy import boundingbox

PointType = np.ndarray  # Tuple[float, float]
PointsType = Sequence[PointType]
PointsList = Sequence[PointsType]
ImageType = np.ndarray

GenericType = TypeVar('GenericType')


def flatten_nest(l: List[List[GenericType]]) -> List[GenericType]:
    '''
    Flatten nested list

    Examples:

    >>> a = [[1, 2], [3, 4, 5]]
    >>> flatten_nest(a)
    [1, 2, 3, 4, 5]
    '''
    return list(itertools.chain.from_iterable(l))


def partition(lst: List[GenericType], sizes: List[int]) -> List[List[GenericType]]:
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
    return [lst[cs[i] : cs[i + 1]] for i in range(len(cs) - 1)]

class ImageDataset(torch.utils.data.Dataset):
    '''
    Dataset for image without points
    '''

    def __init__(self, xs: Sequence[np.ndarray], transform: Optional[Callable[[np.ndarray], np.ndarray]]):
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

    def __init__(
        self,
        x_points: Sequence[Tuple[np.ndarray, PointsList]],
        # xs: Sequence[np.ndarray],
        # points: Sequence[PointsList],
        point_pairs,
        flip_pairs: List[Tuple[int, int]],
        transform: Optional[Callable[[np.ndarray, PointsList, List[Tuple[int, int]]], Tuple[np.ndarray, PointsList]]],
        sigma: float,
    ):
        '''
        Args:
            x_points: list of (x, points).
            point_pairs: pair of 2D points.
            transform: transformation.
            sigma: sigma for heatmap generation
        '''
        # self.xs = xs
        # self.points = points
        self.x_points = x_points
        if point_pairs is None:
            self.point_pairs = [[]] * len(x_points)
        else:
            self.point_pairs = point_pairs
        self.flip_pairs = flip_pairs
        self.transform = transform
        self.sigma = sigma

    def __len__(self):
        return len(self.x_points)

    def __getitem__(self, idx):
        x, points = self.x_points[idx]
        point_pairs = self.point_pairs[idx]
        if self.transform is not None:
            x, points = self.transform(x, points, self.flip_pairs)
        n_chs = len(points) + len(point_pairs)
        y = np.zeros((n_chs, x.shape[1], x.shape[2]), dtype=np.float32)
        for i, p in enumerate(points):
            create_prob_image(x.shape[1:], p, self.sigma, out=y[i])
        # TODO: make affinity map using self.point_pairs
        for i, pairs in enumerate(point_pairs, len(points)):
            raise NotImplementedError

        return x, y


def multivariate_normal(shape: Sequence[int], mu: np.ndarray, sigma: float) -> np.ndarray:
    '''
    Generate PDF(heatmap) for multivariate gaussian

    Args:
        shape (array) : output shape
        mu (array) : mean/center
        sigma (array or scalar): standard deviation
    '''
    if np.isscalar(mu):
        mu = np.repeat(mu, len(shape))
    vecs = [scipy.stats.norm.pdf(np.arange(size), m, sigma) for size, m in zip(shape, mu)]
    # prepare for broadcasting
    vecs = [v.reshape([1 if j != i else s for j in range(len(vecs))]) for i, (v, s) in enumerate(zip(vecs, shape))]
    if len(shape) == 1:
        return vecs[0]

    output = vecs[0]
    for v in vecs[1:]:
        output = output * v
    return output


def create_prob_image(
    shape: Sequence[int],
    points: Sequence[PointType],
    sigma: float,
    normalize: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    '''
    Create heatmap from the given points
    '''
    if len(points) == 0:
        return np.zeros(shape)
    if out is None:
        out = multivariate_normal(shape, points[0], sigma)
    else:
        out[...] = multivariate_normal(shape, points[0], sigma)
    for p in points[1:]:
        np.maximum(out, multivariate_normal(shape, p, sigma), out=out)
    # calculate theoretical max value from sigma
    peak_value = 1 / np.sqrt(2 * np.pi * sigma**2)
    if normalize:
        out *= 1 / peak_value
    assert out is not None  # type narrowing
    return out


# @njit("(i8, i8, f8[:], f8[:], f8, f4[:,:])", cache=True, fastmath=True)
# def _gauss_line_dist(w, h, a, b, sigma, img):
#     a_y, a_x, = a
#     ab = b - a
#     ab_y, ab_x, = ab
#     l = (ab**2).sum()
#     coef = 1.0 / (np.sqrt(2 * np.pi) * sigma)
#     denom = 1.0 / (2 * sigma * sigma)

#     for y in range(h):
#         img_row = img[y]
#         pa_y = y - a_y
#         for x in range(w):
#             pa_x = x - a_x
#             dot = pa_x * ab_x + pa_y * ab_y
#             t = min(max(0.0, dot / l), 1.0)  # clip [0, 1]
#             proj_x = a_x + t * ab_x
#             proj_y = a_y + t * ab_y
#             d_x = x - proj_x
#             d_y = y - proj_y
#             dist_sq = d_x * d_x + d_y * d_y
#             g = coef * np.exp(-dist_sq * denom)
#             img_row[x] = g


# def gauss_line_dist(shape: Sequence[int],
#                     p1: PointType,
#                     p2: PointType,
#                     sigma: float,
#                     normalize: bool = True,
#                     out: Optional[np.ndarray] = None) -> np.ndarray:
#     '''
#     Craete affinity map between two points
#     '''
#     if out is None:
#         out = np.zeros(shape, dtype=np.float32)
#     _gauss_line_dist(shape[1], shape[0], np.array(p1, dtype=np.float64),
#                      np.array(p2, dtype=np.float64), sigma, out)
#     if normalize and out.max() != 0:
#         out *= (1 / out.max())
#     assert out is not None  # type anrrowing
#     return out


def extract_center_of_masses(heatmaps):
    '''
    Center of mass of pixels with the maximum value

    Examples:

    >>> a = np.zeros((2, 3, 3))
    >>> a[0, 1, 1] = 1
    >>> a[1, 1, 2] = 2
    >>> a[1, 1, 1] = 1
    >>> extract_center_of_masses(a)
    [(1.0, 1.0), (1.0, 2.0)]
    '''
    keypoints = []
    for ch in heatmaps:
        max_value = np.max(ch)
        if max_value <= 0:
            raise RuntimeError('Empty heatmap')
        center = center_of_mass(ch == max_value)
        keypoints.append(center)
    return keypoints


def local_center_of_masses(heatmaps, min_thresh=0.25):
    '''

    Note. Channel axis is 0.

    Examples:

    >>> a = np.zeros((1, 5, 5))
    >>> a[0, 1, 1] = 1
    >>> a[0, 3, 4] = 1
    >>> local_center_of_masses(a)
    [[(1.0, 1.0), (3.0, 4.0)]]
    '''
    local_centers_per_ch = []
    for ch in heatmaps:
        max_value = np.max(ch)
        if max_value <= 0:
            local_centers = [[]]
            # raise RuntimeError('Empty heatmap')
        else:
            bbox = boundingbox.bbox(ch > 0)
            # bbox_w_ch = (np.insert(bbox[0], [0], 0), np.insert(bbox[1], [0], heatmaps.shape[0]-1)) # no cropping in channel axis
            margin = 1
            ch = boundingbox.crop(ch, bbox, margin=margin)
            # thresh = max(min_thresh, max_value / 2)
            thresh = min_thresh
            all_labels = measure.label(ch >= thresh)
            local_centers = []
            for i in range(1, np.max(all_labels) + 1):
                island = ch.copy()
                island[all_labels != i] = ch.min()
                local_max = island.max()
                center = center_of_mass(island == local_max)
                center = np.array(center) + bbox[0] - margin
                local_centers.append(center)
        local_centers_per_ch.append(local_centers)
    return local_centers_per_ch


def create_dist_image(shape_: Sequence[int], points_: PointsType, margin: int) -> np.ndarray:
    '''
    Create distance map from the given points

    ```
    dist = create_dist_image((512, 512), [[128, 128], [
    128,
    156,
    ], [0, 128]], 20)
    ```
    '''
    if len(points_) == 0:
        return np.zeros(shape_)
    shape: np.ndarray = np.array(shape_)
    points: np.ndarray = np.array(points_)
    top_left = points.min(axis=0)
    btm_right = points.max(axis=0)
    top_left = np.clip(top_left - margin, 0, None)
    btm_right = np.minimum(btm_right + margin, shape)
    src = np.ones(btm_right - top_left)
    roi_points = points - top_left
    src[roi_points[:, 0], roi_points[:, 1]] = 0
    dist = distance_transform_edt(src)
    assert isinstance(dist, np.ndarray)  # for type checking
    dist = dist.astype(np.float32)
    out = np.full(shape, dist.max(), dtype=np.float32)
    out[top_left[0] : btm_right[0], top_left[1] : btm_right[1]] = dist
    return out
