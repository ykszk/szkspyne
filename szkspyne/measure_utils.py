#%%
import numpy as np
from sympy.geometry import Line, intersection
from sympy.geometry import Point, Line, Polygon, Segment, Circle, intersection
import io
from base64 import b64encode


def img2b64(img):
    with io.BytesIO() as bio:
        img.save(bio, format=img.format)
        # img.save(bio, format='png')
        binary = bio.getvalue()
    b64 = b64encode(binary)
    str_b64 = b64.decode('ascii')
    return str_b64


def corners2midpoints(corners: np.ndarray) -> np.ndarray:
    assert corners.shape == (4, 2)
    midpoints = corners[[0, 1]].mean(axis=0), corners[[
        2, 3
    ]].mean(axis=0), corners[[0, 2]].mean(axis=0), corners[[1, 3]].mean(axis=0)
    return midpoints


def perp(a: np.array) -> np.array:
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b / np.linalg.norm(b)


def rotate_around(pts, angle_rad, origin):
    s, c = np.sin(angle_rad), np.cos(angle_rad)
    r = np.array(((c, -s), (s, c)))
    t = origin
    return r @ (pts - t) + t


def calc_cobb(sup_cnrs: np.ndarray, inf_cnrs: np.ndarray) -> float:
    '''Calculate Cobb angle in degrees
    '''
    up_line = Line(sup_cnrs[0], sup_cnrs[1])
    low_line = Line(inf_cnrs[0], inf_cnrs[1])
    up_low_cross = tuple(intersection(up_line, low_line)[0].evalf())
    angle_rad = float(up_line.angle_between(low_line).evalf())

    if up_low_cross[0] < sup_cnrs[0, 0]:
        pass
    else:
        angle_rad *= -1
    return np.rad2deg(angle_rad)


def calc_intersection(A: np.array, B: np.array, C: np.array,
                      D: np.array) -> np.array:
    '''

    https://stackoverflow.com/a/57821199
    '''
    # a1x + b1y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1 * (A[0]) + b1 * (A[1])

    # a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2 * (C[0]) + b2 * (C[1])

    # determinant
    det = a1 * b2 - a2 * b1

    # parallel line
    if det == 0:
        return (float('inf'), float('inf'))

    # intersect point(x,y)
    x = ((b2 * c1) - (b1 * c2)) / det
    y = ((a1 * c2) - (a2 * c1)) / det
    return np.array((x, y))


def calc_centroid(corners: np.ndarray) -> np.ndarray:
    midpoints = corners2midpoints(corners)
    return calc_intersection(*midpoints)


from collections import defaultdict


def _get_shape_dict(json_data, shape_type):
    # points = defaultdict(list)
    # for shape in json_data['shapes']:
    #     label = shape['label']
    #     if shape['shape_type'] == shape:
    #         points[label].append(shape['points'])
    # return points
    points = defaultdict(list)
    for shape in json_data['shapes']:
        label = shape['label']
        if shape['shape_type'] == shape_type:
            points[label].append(shape['points'])
    return points


def get_point_dict(json_data):
    points = defaultdict(list)
    for shape in json_data['shapes']:
        label = shape['label']
        if shape['shape_type'] == 'point':
            points[label].append(shape['points'][0])
    return points


def get_circle_dict(json_data):
    return _get_shape_dict(json_data, 'circle')


def check_counts(kps):
    assert len(kps['TL']) == len(kps['TR'])
    assert len(kps['BL']) == len(kps['BR'])
    assert len(kps['TL']) == (len(kps['BL']) + 1)


def sort_kps(points):
    kps = {
        k: np.array(sorted(v, key=lambda e: e[1]))
        for k, v in points.items()
    }
    return kps


def get_vert_corners(kps):
    n_vertebrae = len(kps['BR'])
    sacrum_tltr = np.array((kps['TL'][-1], kps['TR'][-1]))
    vert_corners = np.array(
        [kps[l][:n_vertebrae] for l in ['TL', 'TR', 'BL', 'BR']]).transpose(
            (1, 0, 2))
    disc_corners = np.array([
        kps[l][(1 if l in ['TL', 'TR'] else 0):]
        for l in ['TL', 'TR', 'BL', 'BR']
    ]).transpose((1, 0, 2))
    return sacrum_tltr, vert_corners, disc_corners