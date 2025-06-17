import argparse
from typing import List
from szkspyne import settings, labelme_utils
from logzero import logger
from pathlib import Path
import copy
import numpy as np
from scipy.interpolate import CubicSpline
import json

def extrapolate(pts: np.ndarray):
    '''
    Append one extra point using extrapolation
    '''
    pts = pts[pts[:, 1].argsort()]  # sort by y
    bc_type = 'natural'

    xs = np.arange(len(pts))
    sp_x = CubicSpline(xs, pts[:, 0], bc_type=bc_type)
    sp_y = CubicSpline(xs, pts[:, 1], bc_type=bc_type)
    tl_xs = sp_x(len(pts))
    tl_ys = sp_y(len(pts))
    new_pts = np.zeros((pts.shape[0] + 1, 2), dtype=pts.dtype)
    new_pts[:-1] = pts
    new_pts[-1, 0] = tl_xs
    new_pts[-1, 1] = tl_ys
    return new_pts


# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Augment spine point dataset')
parser.add_argument(
    'input',
    help='Input ndjson filename',
    metavar='<input>',
)
parser.add_argument('--mirror', help='Add horizontally mirrored point set to the dataset', action='store_true')
# parser.add_argument('--labelme', help='Input is in labelme format', action='store_true')

args, _unknown = parser.parse_known_args()

if args.input=='-':
    # read from stdin
    logger.info('Reading data from stdin')
    import sys
    all_data = [labelme_utils.ShapeDictLine.model_validate_json(line) for line in sys.stdin]
else:
    all_data: List[labelme_utils.ShapeDictLine] = labelme_utils.ShapeDictLine.from_ndjson(args.input)
logger.info('Original data: %d', len(all_data))
if args.mirror:
    logger.info('Adding horizontally mirrored data')
    LABEL_NAMES = settings.LABEL_NAMES

    mirrored = []
    for data_line in all_data:
        if 'ap_whole' not in data_line.content.flags:
            continue
        data_line = copy.deepcopy(data_line)
        d = data_line.content
        width = d.imageWidth
        kps = d.shapes['point']
        for point in kps.values():
            for p in point:
                p[0][0] = width - p[0][0]
        for flip_pair in settings.FLIP_PAIRS:
            k1, k2 = LABEL_NAMES[flip_pair[0]], LABEL_NAMES[flip_pair[1]]
            if k1 in kps and k2 in kps:
                kps[k1], kps[k2] = kps[k2], kps[k1]
            elif k1 in d and k2 not in d:
                kps[k2] = kps.pop(k1)
            elif k1 not in kps and k2 in kps:
                kps[k1] = kps.pop(k2)
        mirrored.append(data_line)

    all_data += mirrored

VERT_LABELS = settings.VERTEBRA_LABEL_NAMES
L5_POINT_COUNTS = [19, 19, 18, 18]
L4_POINT_COUNTS = (np.array(L5_POINT_COUNTS) - 1).tolist()
L6_POINT_COUNTS = (np.array(L5_POINT_COUNTS) + 1).tolist()


# preprocess
for data_line in all_data:
    d = data_line.content
    ns = {k: len(v) for k, v in d.shapes['point'].items()}
    np_pts = {}
    for label, pts in d.shapes['point'].items():
        if label in VERT_LABELS:
            pts.sort(key=lambda v: v[0][1])  # sort by y
        else:
            pts.sort(key=lambda v: v[0][0])  # sort by x
        np_pts[label] = np.array(pts)[:, 0]  # squeeze (N, 1, 2) -> (N, 2)
    d.shapes['point'] = np_pts

# Create L4 data from L5 (and L5 from L6)
logger.info('Creating L4 data from L5')
l4_data = copy.deepcopy(all_data)
for data_line in l4_data:
    d = data_line.content
    np_pts = {}
    for label, pts in d.shapes['point'].items():
        if label in VERT_LABELS:
            np_pts[label] = pts[1:].copy()
        else:
            np_pts[label] = pts.copy()
    d.shapes['point'] = np_pts

# Create L6 data from L5 (and L5 from L4)
logger.info('Creating L6 data from L5')
l6_data = copy.deepcopy(all_data)
for data_line in l6_data:
    d = data_line.content
    np_pts = {}
    for label, pts in d.shapes['point'].items():
        if label in VERT_LABELS:
            np_pts[label] = extrapolate(pts)
        else:
            np_pts[label] = pts.copy()
    d.shapes['point'] = np_pts
all_data += l4_data + l6_data

logger.info('Total data after augmentation: %d', len(all_data))
valid_count = 0
for data_line in all_data:
    d = data_line.content
    ns = {k: len(v) for k, v in d.shapes['point'].items()}
    valid = True

    for point_count in (L4_POINT_COUNTS, L5_POINT_COUNTS, L6_POINT_COUNTS):
        valid = True
        for label in VERT_LABELS:
            if ns[label] != point_count[VERT_LABELS.index(label)]:
                valid = False
                break

        if valid:
            valid_count += 1
            shapes_point = data_line.content.shapes['point']
            # numpy array to list
            new_points = {}
            for label, pts in shapes_point.items():
                new_points[label] = pts[:, np.newaxis].tolist()
            data_line.content.shapes['point'] = new_points
            lm_data_line = data_line.into_labelme_line()
            # print(data_line.model_dump())
            print(json.dumps(lm_data_line.model_dump(), ensure_ascii=False))
            break

logger.info('Valid data count: %d', valid_count)