import csv
import io
import itertools
from base64 import b64encode
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import svgwrite
from PIL import Image
from scipy import stats
from skspatial.objects import Line, Point

from szkspyne import settings


def pearsonr_ci(x, y, alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x, y)
    r_z = np.arctanh(r)
    se = 1 / np.sqrt(x.size - 3)
    z = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi


def img2b64(img: Image.Image):
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
    return np.array(midpoints)


def lerp(a, b, t):
    if t > 1 or 1 < 0:
        raise ValueError('Invalid value for lerp: %f', t)
    return t * a + (1 - t) * b


def perp(a: np.ndarray) -> np.ndarray:
    '''
    perpendicular unit vector
    '''
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b / np.linalg.norm(b)


def rotate_around(pts, angle_rad, origin):
    s, c = np.sin(angle_rad), np.cos(angle_rad)
    r = np.array(((c, -s), (s, c)))
    t = origin
    return r @ (pts - t) + t


def calc_angle(line1, line2):
    '''
    >>> np.round(np.rad2deg(calc_angle(((0, 0), (1, 1)), ((0, 0), (0, 1)))), 3)
    -45.0
    >>> np.round(np.rad2deg(calc_angle(((0, 0), (-1, -1)), ((0, 0), (0, -1)))), 3)
    45.0
    '''
    line1, line2 = np.array(line1), np.array(line2)
    v1 = line1[1] - line1[0]
    v1 = v1 / np.linalg.norm(v1)
    v2 = line2[1] - line2[0]
    v2 = v2 / np.linalg.norm(v2)
    cos = np.clip(v1 @ v2, -1, 1)
    a = np.arccos(cos)
    if v1[1] - v2[1] <= 0:
        a = -a
    return a


def calc_intersection(A: np.ndarray, B: np.ndarray, C: np.ndarray,
                      D: np.ndarray) -> np.ndarray:
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
        return np.array((float('inf'), float('inf')))

    # intersect point(x,y)
    x = ((b2 * c1) - (b1 * c2)) / det
    y = ((a1 * c2) - (a2 * c1)) / det
    return np.array((x, y))


def calc_centroid(corners: np.ndarray) -> np.ndarray:
    '''
    Intersection of the midpoints
    '''
    midpoints = corners2midpoints(corners)
    return calc_intersection(*midpoints)


def _get_shape_dict(json_data,
                    shape_type) -> Dict[str, List[List[List[float]]]]:
    shapes = defaultdict(list)
    for shape in json_data['shapes']:
        label = shape['label']
        if shape['shape_type'] == shape_type:
            shapes[label].append(shape['points'])
    return dict(shapes)


def get_point_dict(json_data) -> Dict[str, List[List[float]]]:
    '''
    return Dict[label, points]
    '''
    shapes = _get_shape_dict(json_data, 'point')
    points = {}
    for key in shapes.keys():
        points[key] = [s[0] for s in shapes[key]]
    return points


def get_circle_dict(json_data):
    return _get_shape_dict(json_data, 'circle')


def sort_kps(points: Dict[str, Any]) -> Dict[str, np.ndarray]:
    '''
    Sort keypoints by y position

    Args:
    - points Dict[str, List[(x, y)]]

    returns Dict[str, np.ndarray]
    '''
    kps = {
        k: np.array(sorted(v, key=lambda e: e[1]))
        for k, v in points.items()
    }
    return kps


def get_vert_corners(kps):
    n_vertebrae = len(kps['BR'])
    sacrum_tltr = np.array((kps['TL'][-1], kps['TR'][-1]))
    vert_corners = np.array([
        kps[l][:n_vertebrae] for l in settings.VERTEBRA_LABEL_NAMES
    ]).transpose((1, 0, 2))
    return sacrum_tltr, vert_corners


def get_vert_and_disc_corners(kps):
    n_vertebrae = len(kps['BR'])
    sacrum_tltr = np.array((kps['TL'][-1], kps['TR'][-1]))
    vert_corners = np.array([
        kps[l][:n_vertebrae] for l in settings.VERTEBRA_LABEL_NAMES
    ]).transpose((1, 0, 2))
    disc_corners = np.array([
        kps[l][(1 if l in ['TL', 'TR'] else 0):]
        for l in settings.VERTEBRA_LABEL_NAMES
    ]).transpose((1, 0, 2))
    return sacrum_tltr, vert_corners, disc_corners


def read_default_line_colors() -> Dict[str, str]:
    color_filename = Path(__file__).parent / 'default_line_colors.csv'
    color_dict = {}
    with open(color_filename) as f:
        reader = csv.DictReader(f)
        for r in reader:
            color_dict[r['name']] = r['color']
    return color_dict


class SpinalDrawing(svgwrite.Drawing):

    def __init__(self, lm_data: Dict, image_dir: Path, height: int):
        lm_data = lm_data.copy()
        img_fn = Path(image_dir) / Path(lm_data['imagePath']).name
        pixel_size = 1
        original_image_shape = (lm_data['imageWidth'], lm_data['imageHeight'])
        scale = height / original_image_shape[1]
        image_size = (scale * lm_data['imageWidth'] * pixel_size, height * pixel_size)
        super().__init__('output.svg',
                         size=image_size,
                         viewBox=('0 0 {} {}'.format(*image_size)))
        if img_fn.exists():
            img = Image.open(img_fn)
            img.thumbnail(
                (100000,
                height))  # specify height. width argument is just a big number
            w, h = img.size
            image_size = (w * pixel_size, h * pixel_size)
            self.add(
                self.image('data:image/jpg;base64,' + img2b64(img), size=img.size))
        else:
            img = None
            

        for shape in lm_data['shapes']:
            shape['points'] = [[scale * e for e in p] for p in shape['points']]

        text_attrs = dict(stroke='black',
                          stroke_width='1px',
                          fill='white',
                          style="font-size:24px; font-family:sans-serif")

        points = get_point_dict(lm_data)
        kps = sort_kps(points)

        self.size = image_size
        self.text_attrs = text_attrs
        self.points = points
        self.kps = kps
        self.img = img

        if len(kps['TL']) != len(kps['TR']) or len(kps['BL']) != len(
                kps['BR']):
            raise ValueError(
                'Invalid set of number of points for TL, TR, BL and BR')
        sacrum_tltr, vert_corners, disc_corners = get_vert_and_disc_corners(
            kps)
        vert_corners_w_c7 = vert_corners.copy()
        c7_corners = vert_corners[0].copy()
        c7_centroid = calc_centroid(c7_corners)
        self.sacrum_tltr = sacrum_tltr
        self.vert_corners_w_c7 = vert_corners_w_c7
        self.c7_corners = c7_corners
        self.c7_centroid = c7_centroid

        # remove C7
        vert_corners = np.delete(vert_corners, 0, axis=0)
        vert_corners = vert_corners.astype(np.float64)
        disc_corners = np.delete(disc_corners, 0, axis=0)

        self.disc_corners = disc_corners
        self.vert_corners = vert_corners

        sacrum_center = sacrum_tltr.mean(axis=0)
        vert_centroids = np.array([calc_centroid(c) for c in vert_corners])
        disc_centroids = np.array([calc_centroid(c) for c in disc_corners])
        self.sacrum_center = sacrum_center
        self.vert_centroids = vert_centroids
        self.disc_centroids = disc_centroids

        vert_sup_cnrs = vert_corners[:, :2]
        vert_sup_vecs = vert_sup_cnrs[:, 1] - vert_sup_cnrs[:, 0]
        vert_sup_vecs /= np.linalg.norm(vert_sup_vecs, axis=1, keepdims=True)
        vert_inf_cnrs = vert_corners[:, 2:]

        self.vert_sup_cnrs = vert_sup_cnrs
        self.vert_sup_vecs = vert_sup_vecs
        self.vert_inf_cnrs = vert_inf_cnrs

    def draw_line(self,
                  a: np.ndarray,
                  b: np.ndarray,
                  grp,
                  xrange: Union[Tuple[float, float], None] = None,
                  yrange: Union[Tuple[float, float], None] = None):
        d = b - a
        if d[1] == 0: # horizontal line
            p0 = [0, a[1]]
            p1 = [self.size[0], a[1]]
            grp.add(self.path((['M'] + p0 + p1)))
            return p0, p1
        if d[0] == 0: # vertical line
            p0 = [a[0], self.size[0]]
            p1 = [a[0], self.size[1]]
            grp.add(self.path((['M'] + p0 + p1)))
            return p0, p1
        if xrange is None:
            xrange = (0, self.size[0])
        if yrange is None:
            yrange = (0, self.size[1])
        slope = d[1] / d[0]
        y0 = a[1] - slope * a[0]
        intersections = []
        y = slope * xrange[0] + y0
        if yrange[0] <= y <= yrange[1]:
            intersections.append([xrange[0], y])
        y = slope * xrange[1] + y0
        if yrange[0] <= y <= yrange[1]:
            intersections.append([xrange[1], y])
        x = (yrange[0] - y0) / slope
        if xrange[0] <= x <= xrange[1]:
            intersections.append([x, yrange[0]])
        x = (yrange[1] - y0) / slope
        if xrange[0] <= x <= xrange[1]:
            intersections.append([x, yrange[1]])
        p1, p2 = intersections[0], intersections[1]
        grp.add(self.path((['M'] + p1 + p2)))
        return p1, p2

    def draw_cobb(self,
                  sup_cnrs: np.ndarray,
                  inf_cnrs: np.ndarray,
                  grp,
                  aux_dist: float = 3,
                  title: str = '',
                  text_attrs: dict = {}) -> float:
        up_line = Line.from_points(sup_cnrs[0], sup_cnrs[1])
        low_line = Line.from_points(inf_cnrs[0], inf_cnrs[1])
        try:
            up_low_cross = up_line.intersect_line(low_line)
            angle_rad = -calc_angle((sup_cnrs[0], sup_cnrs[1]),
                                    (inf_cnrs[0], inf_cnrs[1]))
            if angle_rad > 0:
                up_end = sup_cnrs[1]
                low_end = inf_cnrs[1]
                pts = up_end + aux_dist * (sup_cnrs[0] - sup_cnrs[1])
            else:
                up_end = sup_cnrs[0]
                low_end = inf_cnrs[0]
                pts = up_end + aux_dist * (sup_cnrs[1] - sup_cnrs[0])

            aux_cross = Point(rotate_around(pts, angle_rad / 2, up_low_cross))
            is_inside = np.linalg.norm(
                aux_cross - sup_cnrs[0]) > np.linalg.norm(up_low_cross -
                                                          sup_cnrs[0])
            if is_inside:
                text_pos = up_low_cross
                grp.add(
                    self.path(
                        (['M'] + up_end.tolist() + up_low_cross.tolist())))
                grp.add(
                    self.path(
                        (['M'] + low_end.tolist() + up_low_cross.tolist())))
            else:
                aux_cross = Point(
                    rotate_around(pts, angle_rad / 2, up_low_cross))
                up_aux1 = up_line.project_point(aux_cross)
                up_aux2 = up_aux1 + 1.5 * (aux_cross - up_aux1)
                up_auxs = np.array((up_aux1, up_aux2))
                low_aux1 = low_line.project_point(aux_cross)
                low_aux2 = low_aux1 + 1.5 * (aux_cross - low_aux1)
                low_auxs = np.array((low_aux1, low_aux2))
                text_pos = aux_cross

                grp.add(
                    self.path((['M'] + up_end.tolist() + up_auxs[0].tolist())))
                grp.add(self.path((['M'] + up_auxs.tolist())))
                grp.add(
                    self.path(
                        (['M'] + low_end.tolist() + low_auxs[0].tolist())))
                grp.add(self.path((['M'] + low_auxs.tolist())))
        except ValueError:
            angle_rad = 0
            text_pos = np.concatenate([sup_cnrs, inf_cnrs],
                                      axis=0).mean(axis=0)
            grp.add(
                self.path(
                    (['M'] + sup_cnrs[0].tolist() + sup_cnrs[1].tolist())))
            grp.add(
                self.path(
                    (['M'] + inf_cnrs[0].tolist() + inf_cnrs[1].tolist())))
            grp.add(
                self.path((['M'] + sup_cnrs.mean(axis=0).tolist() +
                           inf_cnrs.mean(axis=1).tolist())))

        angle_deg = np.rad2deg(angle_rad)
        text_angle = '{:.1f}°'.format(angle_deg)
        text_anchor = 'end' if angle_deg < 0 else 'start'
        text_elm = self.text(text_angle,
                             insert=tuple(text_pos),
                             text_anchor=text_anchor,
                             dominant_baseline="central",
                             **text_attrs)
        if title:
            text_elm.set_desc(title=title)
        grp.add(text_elm)
        return angle_deg

    def draw_vertical_dist(self, p1, p2, grp, title=''):
        grp.add(self.path((['M'] + [0, p1[1]] + [self.size[0], p1[1]])))
        grp.add(self.path((['M'] + [0, p2[1]] + [self.size[0], p2[1]])))
        height_text = '{:.1f} pixels'.format(np.abs(p1[1] - p2[1]))
        text_elm = self.text(height_text, insert=p1, **self.text_attrs)
        if title:
            text_elm.set_desc(title=title)
        grp.add(text_elm)
