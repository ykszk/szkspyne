#%%
import os
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from settings import LABEL_NAMES


def kps2labelme(kps, image_path, image_hw, label_names) -> dict:
    json_data = {
        "version": "4.5.7",
        "flags": {},
        "shapes": [],
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_hw[0],
        "imageWidth": image_hw[1]
    }
    shapes = json_data['shapes']
    assert len(label_names) == len(kps)
    for kps, label in zip(kps, label_names):
        for kp in kps:
            shapes.append({
                "label": label,
                "points": [[kp[1], kp[0]]],
                "group_id": None,
                "shape_type": "point",
                "flags": {}
            })
    return json_data


def labelme2ndjson(input_dir, output, n_jobs):
    input_dir = Path(input_dir)

    def do(fn):
        with open(fn) as f:
            data = json.load(f)
        data['filename'] = fn.name
        data['mtime'] = fn.stat().st_mtime
        l = json.dumps(data)
        return l

    lines = Parallel(n_jobs=n_jobs)(delayed(do)(fn)
                                    for fn in tqdm(input_dir.glob('*.json')))
    with open(output, 'w') as f:
        f.write('\n'.join(lines))


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
