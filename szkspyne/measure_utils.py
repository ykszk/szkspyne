#%%
import numpy as np
import io
from base64 import b64encode
import svgwrite
from collections import defaultdict
from skspatial.objects import Line, Point
from scipy import stats
from szkspyne import settings

def pearsonr_ci(x,y,alpha=0.05):
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

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

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
    return np.array(midpoints)


def perp(a: np.ndarray) -> np.ndarray:
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
    cos = v1 @ v2
    a = np.arccos(cos)
    if v1[1]-v2[1] <= 0:
        a = -a
    return a


def draw_cobb(sup_cnrs: np.ndarray,
              inf_cnrs: np.ndarray,
              dwg: svgwrite.Drawing,
              group_id: str,
              color: str,
              aux_dist: float = 3,
              text_attrs: dict = {},
              hide=False,
              grp=None) -> svgwrite.container.Group:
    up_line = Line.from_points(sup_cnrs[0], sup_cnrs[1])
    low_line = Line.from_points(inf_cnrs[0], inf_cnrs[1])
    view_box = np.array([float(e) for e in dwg['viewBox'].split(' ')]).reshape(2, 2) # [[xmin, ymin], [xmax, ymax]]
    try:
        up_low_cross = up_line.intersect_line(low_line)
        angle_rad = -calc_angle((sup_cnrs[0], sup_cnrs[1]), (inf_cnrs[0], inf_cnrs[1]))
        if angle_rad > 0:
            up_end = sup_cnrs[1]
            low_end = inf_cnrs[1]
            pts = up_end + aux_dist * (sup_cnrs[0] - sup_cnrs[1])
        else:
            up_end = sup_cnrs[0]
            low_end = inf_cnrs[0]
            pts = up_end + aux_dist * (sup_cnrs[1] - sup_cnrs[0])
        
        if grp is None:
            grp = dwg.add(
                dwg.g(id=group_id, stroke=color, stroke_width=2.0, fill='none', visibility='hidden' if hide else 'visible'))
        is_inside = np.all(view_box[0] <= up_low_cross) and np.all(up_low_cross <= view_box[1])
        if is_inside:
            text_pos = up_low_cross
            grp.add(dwg.path((['M'] + up_end.tolist() + up_low_cross.tolist())))
            grp.add(dwg.path((['M'] + low_end.tolist() + up_low_cross.tolist())))
        else:
            aux_cross = Point(rotate_around(pts, angle_rad / 2, up_low_cross))
            up_aux1 = up_line.project_point(aux_cross)
            up_aux2 = up_aux1 + 1.5 * (aux_cross - up_aux1)
            up_auxs = np.array((up_aux1, up_aux2))
            low_aux1 = low_line.project_point(aux_cross)
            low_aux2 = low_aux1 + 1.5 * (aux_cross - low_aux1)
            low_auxs = np.array((low_aux1, low_aux2))
            text_pos = aux_cross

            grp.add(dwg.path((['M'] + up_end.tolist() + up_auxs[0].tolist())))
            grp.add(dwg.path((['M'] + up_auxs.tolist())))
            grp.add(dwg.path((['M'] + low_end.tolist() + low_auxs[0].tolist())))
            grp.add(dwg.path((['M'] + low_auxs.tolist())))
    except ValueError:
        angle_rad = 0
        text_pos = np.concatenate([sup_cnrs, inf_cnrs], axis=0).mean(axis=0)
        grp = dwg.add(
            dwg.g(id=group_id, stroke=color, stroke_width=2.0, fill='none'))
        grp.add(dwg.path((['M'] + sup_cnrs[0].tolist() + sup_cnrs[1].tolist())))
        grp.add(dwg.path((['M'] + inf_cnrs[0].tolist() + inf_cnrs[1].tolist())))
        grp.add(dwg.path((['M'] + sup_cnrs.mean(axis=0).tolist() + inf_cnrs.mean(axis=1).tolist())))

    angle_deg = np.rad2deg(angle_rad)
    text_angle = '{:.1f}°'.format(angle_deg)
    text_anchor = 'end' if angle_deg < 0 else 'start'
    grp.add(dwg.text(text_angle, insert=tuple(text_pos), text_anchor=text_anchor, dominant_baseline="central",
                     **text_attrs))
    return grp, angle_deg


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
    midpoints = corners2midpoints(corners)
    return calc_intersection(*midpoints)


def _get_shape_dict(json_data, shape_type):
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
        [kps[l][:n_vertebrae] for l in settings.VERTEBRA_LABEL_NAMES]).transpose(
            (1, 0, 2))
    return sacrum_tltr, vert_corners

def get_vert_and_disc_corners(kps):
    n_vertebrae = len(kps['BR'])
    sacrum_tltr = np.array((kps['TL'][-1], kps['TR'][-1]))
    vert_corners = np.array(
        [kps[l][:n_vertebrae] for l in settings.VERTEBRA_LABEL_NAMES]).transpose(
            (1, 0, 2))
    disc_corners = np.array([
        kps[l][(1 if l in ['TL', 'TR'] else 0):]
        for l in settings.VERTEBRA_LABEL_NAMES
    ]).transpose((1, 0, 2))
    return sacrum_tltr, vert_corners, disc_corners