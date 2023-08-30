#%%
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Union, Dict, List, Any

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def conv2shape_dict(json_data: dict) -> dict:
    '''Convert labelme dict to shape_type oriented dict


    Examples:

    >>> lm = {'shapes':[{'shape_type':'point','label':'l1','points':[[1,2]]}, {'shape_type':'circle','label':'l2','points':[[1,2],[3,4]]}]}
    >>> conv2shape_dict(lm)
    {'point': {'l1': [[[1, 2]]]}, 'circle': {'l2': [[[1, 2], [3, 4]]]}}
    '''
    shape_dict = defaultdict(lambda: defaultdict(list))
    for shape in json_data['shapes']:
        shape_dict[shape['shape_type']][shape['label']].append(shape['points'])
    # defaultdict -> dict
    shape_dict = {
        k: {kk: vv
            for kk, vv in v.items()}
        for k, v in shape_dict.items()
    }
    return shape_dict


def kps2shapes(kps, label_names) -> list:
    '''
    Create labelme's `shape` element

    input point format is 'yx' and output format is 'xy'
    '''
    shapes = []
    assert len(label_names) == len(kps)
    for kps, label in zip(kps, label_names):
        for kp in kps:
            if len(kp) >= 2:
                shapes.append({
                    "label": label,
                    "points": [[kp[1], kp[0]]],
                    "group_id": None,
                    "shape_type": "point",
                    "flags": {}
                })
    return shapes


def kps2labelme(kps, image_path, image_hw, label_names, flags={}) -> dict:
    '''
    input point format is 'yx'
    '''
    json_data = {
        "version": "4.5.7",
        "flags": flags,
        "shapes": kps2shapes(kps, label_names),
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_hw[0],
        "imageWidth": image_hw[1]
    }
    return json_data


def remove_shapes(labelme_json, shape_labels: List[str]):
    '''
    Remove shapes with given labels from labelme data
    '''
    shapes = labelme_json['shapes']
    for l in shape_labels:
        shapes = list(
            filter(
                lambda shape: shape['label'] != l or shape['shape_type'] !=
                'point', shapes))
    labelme_json['shapes'] = shapes
    return labelme_json

def command_remove_shapes(input_dir, output_dir, shape_labels: List[str], n_jobs: int):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    def do(fn):
        with open(fn) as f:
            data = json.load(f)
        data = remove_shapes(data, shape_labels)
        with open(output_dir/fn.name, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    Parallel(n_jobs=n_jobs)(delayed(do)(fn)
                                    for fn in tqdm(input_dir.glob('*.json')))



def reset(input_dir, n_jobs: int):
    '''
    Delete flags and shapes
    '''
    input_dir = Path(input_dir)

    def do(fn):
        with open(fn) as f:
            data = json.load(f)
        data['flags'] = {}
        data['shapes'] = []
        with open(fn, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    Parallel(n_jobs=n_jobs)(delayed(do)(fn)
                                    for fn in tqdm(input_dir.glob('*.json')))



def labelme2ndjson(input_dir, output, n_jobs: int):
    input_dir = Path(input_dir)

    def do(fn):
        with open(fn) as f:
            data = json.load(f)
        data['filename'] = fn.name
        data['mtime'] = fn.stat().st_mtime
        l = json.dumps(data, ensure_ascii=False)
        return l

    lines = Parallel(n_jobs=n_jobs)(delayed(do)(fn)
                                    for fn in tqdm(input_dir.glob('*.json')))
    assert lines is not None, 'Empty directory'
    with open(output, 'w') as f:
        f.write('\n'.join(lines))


def split_ndjson(filename, output_dir):
    with open(filename) as f:
        jsons = [json.loads(line) for line in f]
    output_dir = Path(output_dir)
    for data in jsons:
        if 'filename' in data:
            outname = output_dir / Path(data.pop('filename')).name
        else:
            outname = output_dir / Path(data.pop('filepath')).name
        mtime = data.pop('mtime', None)
        with open(outname, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if mtime:
            os.utime(outname, (mtime, mtime))


def load_ndjson(filename: Union[str, Path], encoding=None) -> List[Dict]:
    with open(filename, encoding=encoding) as f:
        return [json.loads(line) for line in f]


def save_ndjson(ndjson: List[Any], filename: Union[str, Path], encoding=None, dump_kwargs={}) -> List[Dict]:
    with open(filename, 'w', encoding=encoding) as f:
        for data in ndjson:
            f.write(json.dumps(data, **dump_kwargs))
            f.write('\n')


def diff_ndjsons(fn1, fn2):
    from logzero import logger

    def load_ndjson(fn, key='filename'):
        with open(fn) as f:
            lines = f.readlines()
        d = {}
        for line in lines:
            data = json.loads(line)
            d[data[key]] = data
        return d

    logger.info('Load %s', fn1)
    d1 = load_ndjson(fn1)
    logger.info('Load %s', fn2)
    d2 = load_ndjson(fn2)
    logger.info('Compare')
    for filename in d1.keys():
        e1, e2 = d1[filename], d2[filename]
        if e1 != e2:
            print(filename)


def sort_kps(points):
    '''Sort keypoints by y position in ascending order

    Args:
        points (Dict[str, points]): keypoints
    '''
    kps = {
        k: np.array(sorted(v, key=lambda e: e[1]))
        for k, v in points.items()
    }
    return kps


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Utility functions for labelme.')
    subparsers = parser.add_subparsers()
    p_ndjson = subparsers.add_parser(
        'ndjson', help="Create ndjson from a labelme directory.")
    p_ndjson.add_argument('input', help='Input directory')
    p_ndjson.add_argument('output', help='Output jsonl file')

    def command_ndjson(args):
        labelme2ndjson(args.input, args.output, len(os.sched_getaffinity(0)))

    p_ndjson.set_defaults(func=command_ndjson)

    p_rm = subparsers.add_parser(
        'remove', help="Remove shapes with given label[s] from labeme files in the labelme directory.")
    p_rm.add_argument('input', help='Input directory')
    p_rm.add_argument('output', help='Output directory')
    p_rm.add_argument('label', help='Shape label[s] to remove', nargs='+')

    def command_rm(args):
        command_remove_shapes(args.input, args.output, args.label, len(os.sched_getaffinity(0)))

    p_rm.set_defaults(func=command_rm)

    p_reset = subparsers.add_parser(
        'reset', help="Reset (empty flags and shapes) labeme files in the labelme directory.")
    p_reset.add_argument('input', help='Input directory')

    def command_reset(args):
        reset(args.input, len(os.sched_getaffinity(0)))

    p_reset.set_defaults(func=command_reset)

    p_split = subparsers.add_parser(
        'split_ndjson',
        help="Split input ndjson and save as json file in the output directory."
    )
    p_split.add_argument('input', help='Input ndjson')
    p_split.add_argument('output', help='Output directory')

    def command_split(args):
        split_ndjson(args.input, args.output)

    p_split.set_defaults(func=command_split)

    p_diff = subparsers.add_parser('diff', help="Compare two ndjsons.")
    p_diff.add_argument('input1', help='Input ndjson')
    p_diff.add_argument('input2', help='Input ndjson')

    def command_diff(args):
        diff_ndjsons(args.input1, args.input2)

    p_diff.set_defaults(func=command_diff)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        print('No command was specified.')
        parser.print_help()
        sys.exit(1)
    return (args.func(args))


if __name__ == '__main__':
    sys.exit(main())
