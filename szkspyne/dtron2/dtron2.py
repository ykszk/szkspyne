#%%
from pathlib import Path
import sys
import argparse
import json
import numpy as np
from tqdm import tqdm
import detectron2.structures
import sys
import argparse
from szkspyne import labelme_utils

# [Use Custom Datasets — detectron2 0.6 documentation](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)

def main():
    parser = argparse.ArgumentParser(description='Detectron2 utility tools.')
    parser.add_argument('input', help='Input labelme ndjson', metavar='<input>')
    parser.add_argument('output', help='Output ndjson filename', metavar='<output>')
    parser.add_argument(
        '--image_dir',
        help='Image directory. default: %(default)s',
        default='../../data/annotation/images/all')

    args = parser.parse_args()

    ndjson_filename = Path(args.input)
    image_dir = Path(args.image_dir)

    with open(ndjson_filename) as f:
        lines = [line for line in f]
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
        MARGIN = 32
        for corners in vert_corners:
            tlx, tly = corners[:, 0].min(), corners[:, 1].min()
            brx, bry = corners[:, 0].max(), corners[:, 1].max()
            keypoints = np.append(corners,
                                np.full([corners.shape[0], 1], VISIBILITY),
                                axis=1)
            anno = {
                'bbox': [tlx-MARGIN, tly-MARGIN, brx+MARGIN, bry+MARGIN],
                'bbox_mode': BBOX_MODE,
                'category_id': CATEGORY_ID,
                'keypoints': keypoints.flatten().tolist()
            }
            dtron2_data['annotations'].append(anno)
        new_dataset.append(dtron2_data)

    with open(args.output/'all.ndjson', 'w') as f:
        for data in new_dataset:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
