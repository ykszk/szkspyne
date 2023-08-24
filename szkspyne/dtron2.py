#%%
from pathlib import Path
import sys
import argparse
import json
import labelme_utils
import numpy as np

import sys
import argparse


# def main():
if True:
    parser = argparse.ArgumentParser(
        description='Detectron2 utility tools.')
    # parser.add_argument('input', help='Input filename', metavar='<input>')
    # parser.add_argument('output', help='Output filename', metavar='<output>')
    parser.add_argument('--box_size',
                        help='Bounding box size for a point. default: %(default)s',
                        type=float,
                        default=64)
    # parser.add_argument('-s',
    #                     '--switch',
    #                     help='Switch argument',
    #                     action='store_true')

    # args = parser.parse_args()
    args, _ = parser.parse_known_args()


# if __name__ == '__main__':
#     sys.exit(main())

ndjson_filename = Path('../script/manual_annotation_wo_CLO.ndjson')

with open(ndjson_filename) as f:
    for line in f:
        original = json.loads(line)
        break

kps = {}
for l in ['TL', 'TR', 'BR', 'BL']:
    kps[l] = np.array(original['point'][l])[:, 0]  # remove redundant dimension
kps = labelme_utils.sort_kps(kps)

# remove sacral points
sacral_tl = kps['TL'][-1]
sacral_tr = kps['TR'][-1]
kps['TL'] = kps['TL'][:-1]
kps['TR'] = kps['TR'][:-1]

if not (len(kps['TL']) == len(kps['TR']) == len(kps['BR']) == len(kps['BL'])):
    raise RuntimeError(
        'Invalid number of keypoints: %d, %d, %d, %d',
        len(kps['TL']), len(kps['TR']), len(kps['BR']), len(kps['BL']))

dtron2_data = {'file_name': original['filepath'], 'height': original['imageHeight'], 'width': original['imageWidth'], 'image_id': original['filepath'], 'annotations': []}

vert_corners = np.stack([kps['TL'], kps['TR'], kps['BL'], kps['BR']]).transpose((1, 0, 2)) # [vert, corner, xy]
assert vert_corners.shape[1] == 4 and vert_corners.shape[2] == 2

BBOX_MODE = 0 # BoxMode.XYXY_ABS
VISIBILITY = 2 # visible
for corners in vert_corners:
    tlx, tly = corners[:, 0].min(), corners[:, 1].min()
    brx, bry = corners[:, 0].max(), corners[:, 1].max()
    keypoints = np.append(corners, np.full([corners.shape[0], 1], VISIBILITY), axis=1)
    anno = {'bbox': [tlx, tly, brx, bry],
            'bbox_mode': BBOX_MODE,
            'category_id': 0,
            'keypoints': keypoints.flatten().tolist()}
    dtron2_data['annotations'].append(anno)

# %%
import settings
settings.NON_VERTEBRA_LABEL_NAMES