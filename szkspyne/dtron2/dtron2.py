#%%
from pathlib import Path
import sys
import argparse
import json
import labelme_utils
import numpy as np
from tqdm import tqdm
import detectron2.structures
import sys
import argparse

# [Use Custom Datasets — detectron2 0.6 documentation](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)

def main():
    parser = argparse.ArgumentParser(description='Detectron2 utility tools.')
    # parser.add_argument('input', help='Input filename', metavar='<input>')
    # parser.add_argument('output', help='Output filename', metavar='<output>')
    # parser.add_argument(
    #     '--box_size',
    #     help='Bounding box size for a point. default: %(default)s',
    #     type=float,
    #     default=64)
    # parser.add_argument('-s',
    #                     '--switch',
    #                     help='Switch argument',
    #                     action='store_true')

    # args = parser.parse_args()
    args, _ = parser.parse_known_args()

# if __name__ == '__main__':
#     sys.exit(main())

ndjson_filename = Path('../../script/lm_manual_annotation.ndjson')
outdir = Path('../../script/tmpdtron2')
image_dir = Path('../../../data/annotation/images/all')

with open(ndjson_filename) as f:
    lines = [line for line in f]
lines = lines[:10] # TODO: remove
new_dataset = []
for line in tqdm(lines):
    original = json.loads(line)

    kps = labelme_utils.conv2shape_dict(original)
    # kps = {}
    for l in ['TL', 'TR', 'BR', 'BL']:
        kps[l] = np.array(kps['point'][l])[:, 0]  # remove redundant dimension
    kps = labelme_utils.sort_kps(kps)

    # remove sacral points
    sacral_tl = kps['TL'][-1]
    sacral_tr = kps['TR'][-1]
    kps['TL'] = kps['TL'][:-1]
    kps['TR'] = kps['TR'][:-1]

    if not (len(kps['TL']) == len(kps['TR']) == len(kps['BR']) == len(kps['BL'])):
        raise RuntimeError('Invalid number of keypoints: %d, %d, %d, %d',
                        len(kps['TL']), len(kps['TR']), len(kps['BR']),
                        len(kps['BL']))

    dtron2_data = {
        'file_name': str(image_dir / original['imagePath']),
        'height': original['imageHeight'],
        'width': original['imageWidth'],
        'image_id': original['imagePath'],
        'annotations': []
    }

    vert_corners = np.stack([kps['TL'], kps['TR'], kps['BL'],
                            kps['BR']]).transpose((1, 0, 2))  # [vert, corner, xy]
    assert vert_corners.shape[1] == 4 and vert_corners.shape[2] == 2

    BBOX_MODE = detectron2.structures.BoxMode.XYXY_ABS
    VISIBILITY = 2  # all points are visible
    CATEGORY_ID = 0
    for corners in vert_corners:
        tlx, tly = corners[:, 0].min(), corners[:, 1].min()
        brx, bry = corners[:, 0].max(), corners[:, 1].max()
        keypoints = np.append(corners,
                            np.full([corners.shape[0], 1], VISIBILITY),
                            axis=1)
        anno = {
            'bbox': [tlx, tly, brx, bry],
            'bbox_mode': BBOX_MODE,
            'category_id': CATEGORY_ID,
            'keypoints': keypoints.flatten().tolist()
        }
        dtron2_data['annotations'].append(anno)
    new_dataset.append(dtron2_data)

with open(outdir/'all.ndjson', 'w') as f:
    for data in new_dataset:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')

# %%
import settings

settings.NON_VERTEBRA_LABEL_NAMES