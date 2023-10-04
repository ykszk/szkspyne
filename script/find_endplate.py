import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.spatial as spatial

from szkspyne import labelme_utils
from szkspyne.utils import measure as measure_utils


parser = argparse.ArgumentParser(description='Find closest end plates.')
parser.add_argument('input', help='Exp result directory', metavar='<input>')

args = parser.parse_args()


result_dir = Path(args.input)
input_filename = result_dir / 'auto/keypoints/ap_whole/all_final_keypoints.ndjson'
input_indices_filename = result_dir / 'auto/measures_ap_whole.ndjson'
ref_points_filename = result_dir / 'reference/keypoints/ap_whole/all_final_keypoints.ndjson'
ref_indices_filename = result_dir / 'reference/measures_ap_whole.ndjson'
output_indices_filename = result_dir / 'auto/closest_indices.ndjson'


def to_named_dict(ndjson: List[Dict]):
    ret = {}
    for obj in ndjson:
        ret[obj['filename']] = obj
    return ret


def load_ndjson(filename: Path):
    return to_named_dict(labelme_utils.load_ndjson(filename))


ref_indices_ndjson = load_ndjson(ref_indices_filename)
ref_points_ndjson = load_ndjson(ref_points_filename)
input_points_ndjson = load_ndjson(input_filename)
input_indices_ndjson = load_ndjson(input_indices_filename)


def obj2vertebrae(obj: dict):
    vert = measure_utils.Vertebrae(obj)
    return vert.vert_corners


d_indices = {}
for file_id, indices_obj in ref_indices_ndjson.items():
    input_obj = input_points_ndjson[file_id]
    curves = indices_obj['measures']['curve_indices']
    input_verts = obj2vertebrae(input_obj).reshape(-1, 2, 2, 2)  # [index, top/bottom, lr, point2d]
    ref_verts = obj2vertebrae(ref_points_ndjson[file_id]).reshape(-1, 2, 2, 2)
    closest_indices = {}
    for label, indices in curves.items():
        if indices is None:
            closest_indices[label] = None
            continue
        closest_indices[label] = []
        for top_bottom, index in enumerate(indices):
            input_points = input_verts[:, top_bottom]
            ref_points = ref_verts[index, top_bottom]
            tl_dist = spatial.distance.cdist(ref_points[[0]], input_points[:, 0])[0]
            tr_dist = spatial.distance.cdist(ref_points[[1]], input_points[:, 1])[0]
            dist = tl_dist + tr_dist
            closest_indices[label].append(int(np.argmin(dist)))
    d_indices[file_id] = {'filename': file_id, 'curve_indices': [closest_indices]}


with open(output_indices_filename, 'w') as f:
    for obj in d_indices.values():
        f.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')))
        f.write('\n')
