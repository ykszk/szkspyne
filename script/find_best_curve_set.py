import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from szkspyne import labelme_utils
from szkspyne.utils import measure as measure_utils


parser = argparse.ArgumentParser(description='Find best curve set.')
parser.add_argument('input', help='Exp result directory', metavar='<input>')

args = parser.parse_args()


result_dir = Path(args.input)
input_indices_filename = result_dir / 'auto/curve_indices_ap_whole.ndjson'
ref_indices_filename = result_dir / 'reference/measures_ap_whole.ndjson'
output_indices_filename = result_dir / 'auto/best_indices.ndjson'


def to_named_dict(ndjson: List[Dict]):
    ret = {}
    for obj in ndjson:
        ret[obj['filename']] = obj
    return ret


def load_ndjson(filename: Path):
    return to_named_dict(labelme_utils.load_ndjson(filename))


ref_indices_ndjson = load_ndjson(ref_indices_filename)
input_indices_ndjson = load_ndjson(input_indices_filename)
assert len(ref_indices_ndjson) == len(input_indices_ndjson)


def obj2vertebrae(obj: dict):
    vert = measure_utils.Vertebrae(obj)
    return vert.vert_corners


d_indices = {}
for file_id, ref_indices_obj in ref_indices_ndjson.items():
    in_indices_obj = input_indices_ndjson[file_id]
    ref_measures = ref_indices_obj['measures']
    ref_indices = ref_measures['curve_indices']
    all_alt_measures = in_indices_obj['curve_indices']
    best_indices = all_alt_measures[0]
    if len(all_alt_measures) > 1:
        ref_keys = [k for k, v in ref_indices.items() if v is not None]
        scores = []
        for alt_measures in all_alt_measures:
            alt_keys = [k for k, v in alt_measures['indices'].items() if v is not None]
            common_keys = set(ref_keys).intersection(set(alt_keys))
            score = 0
            for key in common_keys:
                score += abs(ref_measures[key] - alt_measures['angles'][key])
            scores.append(score)
        best_indices = all_alt_measures[np.argmin(scores)]

    best_indices['name'] = 'Best'
    d_indices[file_id] = {
        'filename': file_id,
        'curve_indices': [best_indices],
    }


with open(output_indices_filename, 'w') as f:
    for obj in d_indices.values():
        f.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')))
        f.write('\n')
