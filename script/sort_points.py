import argparse
import json
import sys

import numpy as np

from szkspyne import labelme_utils


def main():
    parser = argparse.ArgumentParser(description='Sort points in labelme json.')
    parser.add_argument('input', help='Input ndjson filename', metavar='<input>')
    parser.add_argument('-x', help='Sort by x coordinates', action='store_true')
    parser.add_argument('-r', '--reverse', help='Sort in descending order', action='store_true')
    args = parser.parse_args()
    if args.input == '-':
        lines = sys.stdin.readlines()
    else:
        with open(args.input) as f:
            lines = f.readlines()

    sign = -1 if args.reverse else 1
    pos = 1 if args.x else 0
    for line in lines:
        obj_line = json.loads(line)
        obj = obj_line['content']
        shapes = labelme_utils.conv2shape_dict(obj)
        points = shapes['point']
        kps = {k: sorted([[p[0][1], p[0][0]] for p in v], key=lambda e: sign * e[pos]) for k, v in points.items()}
        obj['shapes'] = labelme_utils.kps2shapes(kps.values(), kps.keys())
        print(json.dumps(obj_line))


if __name__ == '__main__':
    sys.exit(main())
