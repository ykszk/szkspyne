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
# output filename
parser.add_argument('--l4', help='Output L4 augmented ndjson filename', metavar='<l4>')
parser.add_argument('--l5', help='Output L5 augmented ndjson filename', metavar='<l5>')
parser.add_argument('--l6', help='Output L6 augmented ndjson filename', metavar='<l6>')
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
        data_line.content.flags.add('mirrored')
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
    # Remove bottom point of vertebrae
    for label, pts in d.shapes['point'].items():
        if label in VERT_LABELS:
            np_pts[label] = pts[:-1].copy()
        else:
            np_pts[label] = pts.copy()
    # # reset "C7-TL" and "C7-TR" with highest point (minimum y) of TL and TR respectively
    # if 'TL' in d.shapes['point'] and 'C7-TL' in d.shapes['point']:
    #     tl_pts = d.shapes['point']['TL']
    #     min_idx = np.argmin(tl_pts[:, 1])
    #     np_pts['C7-TL'] = tl_pts[min_idx:min_idx+1].copy()
    # if 'TR' in d.shapes['point'] and 'C7-TR' in d.shapes['point']:
    #     tr_pts = d.shapes['point']['TR']
    #     min_idx = np.argmin(tr_pts[:, 1])
    #     np_pts['C7-TR'] = tr_pts[min_idx:min_idx+1].copy()
    # reset "S-TL" and "S-TR" with lowest point (maximum y) of TL and TR respectively
    if 'TL' in d.shapes['point'] and 'S-TL' in d.shapes['point']:
        tl_pts = np_pts['TL']
        max_idx = np.argmax(tl_pts[:, 1])
        np_pts['S-TL'] = tl_pts[max_idx:max_idx+1].copy()
    if 'TR' in d.shapes['point'] and 'S-TR' in d.shapes['point']:
        tr_pts = np_pts['TR']
        max_idx = np.argmax(tr_pts[:, 1])
        np_pts['S-TR'] = tr_pts[max_idx:max_idx+1].copy()
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
    # reset "S-TL" and "S-TR" with lowest point (maximum y) of TL and TR respectively
    if 'TL' in d.shapes['point'] and 'S-TL' in d.shapes['point']:
        tl_pts = np_pts['TL']
        max_idx = np.argmax(tl_pts[:, 1])
        np_pts['S-TL'] = tl_pts[max_idx:max_idx+1].copy()
    if 'TR' in d.shapes['point'] and 'S-TR' in d.shapes['point']:
        tr_pts = np_pts['TR']
        max_idx = np.argmax(tr_pts[:, 1])
        np_pts['S-TR'] = tr_pts[max_idx:max_idx+1].copy()
    d.shapes['point'] = np_pts
all_data += l4_data + l6_data

logger.info('Total data after augmentation: %d', len(all_data))
valid_count = 0
lines = {'L4': [], 'L5': [], 'L6': []}
for data_line in all_data:
    d = data_line.content
    ns = {k: len(v) for k, v in d.shapes['point'].items()}
    valid = True

    for l_count, point_count in zip(['L4', 'L5', 'L6'], [L4_POINT_COUNTS, L5_POINT_COUNTS, L6_POINT_COUNTS]):
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
            flags = lm_data_line.content.flags
            for l in ['L4', 'L5', 'L6']:
                flags.pop(l, None)
            flags[l_count] = True
            lines[l_count].append(lm_data_line)
            break

logger.info('Valid data count: %d', valid_count)
if args.l4:
    labelme_utils.ShapeDictLine.save_to_ndjson(lines['L4'], args.l4)
    logger.info('Wrote L4 data (%d) to %s', len(lines['L4']), args.l4)
if args.l5:
    labelme_utils.ShapeDictLine.save_to_ndjson(lines['L5'], args.l5)
    logger.info('Wrote L5 data (%d) to %s', len(lines['L5']), args.l5)
if args.l6:
    labelme_utils.ShapeDictLine.save_to_ndjson(lines['L6'], args.l6)
    logger.info('Wrote L6 data (%d) to %s', len(lines['L6']), args.l6)