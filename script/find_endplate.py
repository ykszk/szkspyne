# %%
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.spatial as spatial

from szkspyne import labelme_utils

parser = argparse.ArgumentParser(description='Find closest end plates.')
parser.add_argument('input', help='Exp result directory', metavar='<input>')

args = parser.parse_args()


result_dir = Path(args.input)
# result_dir = Path('/poseidon/users/suzuki/projects/scoliosis/result/exp20250109')
reference_filename = result_dir / 'reference/curve_and_points.ndjson'
ai_filename = result_dir / 'auto/keypoints/ap_whole/no-bend_final_keypoints_native.ndjson'
output_filename = result_dir / 'closest/no-bend_final_keypoints_native.ndjson'


def to_named_dict(ndjson: List[Dict]):
    ret = {}
    for obj in ndjson:
        ret[Path(obj['filename']).name] = obj
    return ret


def load_ndjson(filename: Path):
    return to_named_dict(labelme_utils.load_ndjson(filename))


# ref_indices_ndjson = load_ndjson(ref_indices_filename)
# ref_points_ndjson = load_ndjson(ref_points_filename)
ref_points_ndjson = load_ndjson(reference_filename)
ai_points_ndjson = load_ndjson(ai_filename)
# input_indices_ndjson = load_ndjson(input_indices_filename)

# %%
from szkspyne.curve import CurveInfo, Curve
from szkspyne.utils.measure import VertebraDiscIndex
import copy

results = []
for file_id, ref_obj in ref_points_ndjson.items():
    ref_points = ref_obj['content']['spine']
    ref_points = np.array(ref_points[1:-1])  # exclude cervical and sacral points
    curve_info = CurveInfo.model_validate(ref_obj['content'])
    ai_data = ai_points_ndjson[file_id]
    ai_points = ai_data['content']['spine']
    ai_points = np.array(ai_points[1:-1])  # exclude cervical and sacral points
    new_curve_info = copy.deepcopy(curve_info)
    for curve_name in ['PT', 'MT', 'TLL']:
        curve = getattr(curve_info.curves, curve_name)
        if curve is None:
            continue
        sup_points = ref_points[curve[0].sup]
        left_dists = spatial.distance.cdist(sup_points[[0]], ai_points[:, 0])
        right_dists = spatial.distance.cdist(sup_points[[1]], ai_points[:, 1])
        dists = left_dists + right_dists
        sup_closest_index = np.argmin(dists)
        inf_points = ref_points[curve[0].inf]
        left_dists = spatial.distance.cdist(inf_points[[2]], ai_points[:, 2])
        right_dists = spatial.distance.cdist(inf_points[[3]], ai_points[:, 3])
        dists = left_dists + right_dists
        inf_closest_index = np.argmin(dists)
        new_curve = Curve(sup=int(sup_closest_index), inf=int(inf_closest_index))
        setattr(new_curve_info.curves, curve_name, (new_curve, None))
        # adjust apex
        ref_apex = getattr(curve_info.apices, curve_name)
        new_apex = VertebraDiscIndex(min(ref_apex, 2 * (len(ai_points) - 1)))  # apex numbering includes discs
        setattr(new_curve_info.apices, curve_name, new_apex)
    ai_data['content'].update(new_curve_info.model_dump())
    results.append(ai_data)
# %%
output_filename.parent.mkdir(parents=True, exist_ok=True)
with open(output_filename, 'w') as f:
    for ref_obj in results:
        f.write(json.dumps(ref_obj, ensure_ascii=False, separators=(',', ':')))
        f.write('\n')

# %%
