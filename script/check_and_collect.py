# %%
from pathlib import Path
import json
from collections import defaultdict
from typing import List, Union
import numpy as np
import tqdm
import argparse

parser = argparse.ArgumentParser(
    description='Create ndjson from a labelme directory.')
parser.add_argument('input', help='Input data directory', metavar='<input>')
parser.add_argument('output', help='Output directory', metavar='<output>')
parser.add_argument('--annotators',
                    help='List of annotators. default: %(default)s',
                    metavar='<list>',
                    default=['藤森', '中嶋', '蟹江'])
args = parser.parse_args()

data_dir = Path(args.input)
annotators = set(args.annotators)
outdir = Path(args.output)


def save_ndjson(filename: Union[Path, str], json_list: List[dict]):
    with open(filename, 'w') as f:
        for j in json_list:
            json.dump(j, f, ensure_ascii=False)
            f.write('\n')


for kind in ['ap_whole', 'lateral_whole']:
    # for kind in ['ap_whole']:
    fns = sorted(data_dir.glob(kind + '/*.json'))
    valid_fns = []
    valid_json = []
    invalid_fns = []
    invalid_json = []
    for fn in tqdm.tqdm(fns):
        with open(fn) as f:
            json_data = json.load(f)
        json_data['filepath'] = str(fn)
        json_flags = json_data['flags']
        assert isinstance(json_flags, dict)
        flags = set((k for k, i in json_flags.items() if i))
        annotator = annotators.intersection(flags)
        if len(annotator) == 0:
            continue
        if False:
            if len(annotator) > 1:
                raise RuntimeError('More than one annotators' + str(annotator))
            annotator = list(annotator)[0]
        point_dict = defaultdict(list)
        for shape in json_data['shapes']:
            if shape['shape_type'] == 'point':
                point_dict[shape['label']].append(shape['points'][0])
        point_dict = {k: np.array(i) for k, i in point_dict.items()}
        try:
            assert len(point_dict['TL']) == len(point_dict['TR']), 'TL!=TR'
            assert len(point_dict['BL']) == len(point_dict['BR']), 'BL!=BR'
            assert len(point_dict['TL']) == (
                len(point_dict['BL']) + 1), 'TL!=(BL+1)'
        except Exception as e:
            print(fn, e)
            invalid_fns.append(fn)
            invalid_json.append(json_data)
            continue
        valid_fns.append(fn)
        valid_json.append(json_data)
    print('Valid:', len(fns), len(valid_fns))
    print('Invalid:', len(invalid_json))
    outname = outdir / ('valid_' + kind+'.ndjson')
    outname.parent.mkdir(parents=True, exist_ok=True)
    save_ndjson(outname, valid_json)
    if len(invalid_json) > 0:
        outname = outdir / ('invalid_' + kind+'.ndjson')
        save_ndjson(outname, invalid_json)
