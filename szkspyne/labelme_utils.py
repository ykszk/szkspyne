# %%
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Union, Callable

import numpy as np
from joblib import Parallel, delayed
from logzero import logger
from pydantic import BaseModel
from tqdm import tqdm
from szkspyne.type import LineMixin

class LabelMe(BaseModel):
    class Shape(BaseModel):
        label: str
        points: List[List[float]]
        group_id: Optional[int]
        shape_type: str
        flags: Dict[str, bool]

        @staticmethod
        def from_dict(data: Dict[str, Any]) -> 'LabelMe.Shape':
            return LabelMe.Shape(
                label=data['label'],
                points=data['points'],
                group_id=data.get('group_id'),
                shape_type=data['shape_type'],
                flags=data.get('flags', {}),
            )

    version: str
    shapes: List[Shape]
    imagePath: str
    imageData: Optional[str]
    imageHeight: int
    imageWidth: int
    flags: Dict[str, bool]

    @staticmethod
    def from_file(filename: Union[str, Path]) -> 'LabelMe':
        with open(filename) as f:
            return LabelMe.model_validate_json(f.read())

    def shift(self, tx: float, ty: float) -> "LabelMe":
        shapes = []
        for shape in self.shapes:
            points = [[p[0] + tx, p[1] + ty] for p in shape.points]
            shapes.append(shape.model_copy(update={'points': points}))
        return self.model_copy(update={'shapes': shapes})

    def scale(self, s: float) -> "LabelMe":
        shapes = []
        for shape in self.shapes:
            points = [[p[0] * s, p[1] * s] for p in shape.points]
            shapes.append(shape.model_copy(update={'points': points}))
        return self.model_copy(update={'shapes': shapes})
    
    def rotate(self, angle_deg: float, center: Optional[List[float]] = None) -> "LabelMe":
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        if center is None:
            center = [self.imageWidth / 2, self.imageHeight / 2]

        cx, cy = center
        shapes = []
        for shape in self.shapes:
            points = []
            for p in shape.points:
                x_shifted = p[0] - cx
                y_shifted = p[1] - cy
                x_rotated = x_shifted * cos_a - y_shifted * sin_a + cx
                y_rotated = x_shifted * sin_a + y_shifted * cos_a + cy
                points.append([x_rotated, y_rotated])
            shapes.append(shape.model_copy(update={'points': points}))
        return self.model_copy(update={'shapes': shapes})

    def _into_dict(self) -> Dict[str, Dict[str, list]]:
        '''
        Dict[shape: str, Dict[label: str, points: list]]
        '''
        shape_dict: DefaultDict[str, DefaultDict[str, list]] = defaultdict(lambda: defaultdict(list))
        for shape in self.shapes:
            shape_dict[shape.shape_type][shape.label].append(shape.points)

        # defaultdict -> dict
        output = {k: {kk: vv for kk, vv in v.items()} for k, v in shape_dict.items()}
        return output

    def into_labelme_line(self, filename: str) -> "LabelMeLine":
        return LabelMeLine(content=self, filename=filename)

    def into_shape_dict(self) -> "ShapeDict":
        shapes = self._into_dict()
        shape_dict = ShapeDict(
            shapes=shapes,
            imagePath=self.imagePath,
            imageHeight=self.imageHeight,
            imageWidth=self.imageWidth,
            flags=self.flag_set(),
        )
        return shape_dict

    def flag_set(self) -> Set[str]:
        return {k for k, v in self.flags.items() if v}


class ShapeDict(BaseModel):
    shapes: Dict[str, Dict[str, list]]
    imagePath: str
    imageHeight: int
    imageWidth: int
    flags: Set[str]

    def into_labelme(self) -> "LabelMe":
        shapes = []
        for shape_type, labels in self.shapes.items():
            for label, points in labels.items():
                for p in points:
                    shapes.append(
                        LabelMe.Shape(
                            label=label,
                            points=p,
                            group_id=None,
                            shape_type=shape_type,
                            flags={},
                        )
                    )
        return LabelMe(
            version="4.5.7",
            shapes=shapes,
            imagePath=self.imagePath,
            imageData=None,
            imageHeight=self.imageHeight,
            imageWidth=self.imageWidth,
            flags={k: True for k in self.flags},
        )
    
    def transform(self, tf: Callable):
        '''
        Apply a transformation to the points in the shape dict.
        
        Args:
            tf (Callable): A function that takes a list of points and returns a transformed list of points.
        '''
        for shape_type, labels in self.shapes.items():
            for label, points in labels.items():
                for i, p in enumerate(points):
                    points[i] = tf(p)
        return self


class ShapeDictLine(BaseModel, LineMixin):
    content: ShapeDict
    filename: str

    @staticmethod
    def from_file(filename: Union[str, Path]) -> 'ShapeDictLine':
        with open(filename) as f:
            return ShapeDictLine.model_validate_json(f.read())

    def into_labelme_line(self) -> "LabelMeLine":
        return LabelMeLine(content=self.content.into_labelme(), filename=self.filename)


class LabelMeLine(BaseModel, LineMixin):
    content: LabelMe
    filename: str

    def into_shape_dict_line(self) -> ShapeDictLine:
        shape_dict = self.content.into_shape_dict()
        return ShapeDictLine(content=shape_dict, filename=self.filename)


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
    shape_dict = {k: {kk: vv for kk, vv in v.items()} for k, v in shape_dict.items()}
    return shape_dict


def kps2shapes(kps, label_names, xy_input=False) -> list:
    '''
    Create labelme's `shape` element

    input point format is 'yx' and output format is 'xy'
    '''
    shapes = []
    if xy_input:
        i_x, i_y = 0, 1
    else:
        i_x, i_y = 1, 0
    assert len(label_names) == len(kps), f'Number of label names ({len(label_names)}) does not match number of keypoints ({len(kps)})'
    for kps, label in zip(kps, label_names):
        for kp in kps:
            if len(kp) >= 2:
                shapes.append(
                    {"label": label, "points": [[kp[i_x], kp[i_y]]], "group_id": None, "shape_type": "point", "flags": {}}
                )
            elif len(kp) == 1 and len(kp[0]) == 2:
                shapes.append(
                    {"label": label, "points": [[kp[0][i_x], kp[0][i_y]]], "group_id": None, "shape_type": "point", "flags": {}}
                )
    return shapes


def kps2labelme(kps, image_path, image_hw, label_names, flags={}) -> LabelMe:
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
        "imageWidth": image_hw[1],
    }
    return LabelMe.model_validate(json_data)


def remove_shapes(labelme_json, shape_labels: List[str]):
    '''
    Remove shapes with given labels from labelme data
    '''
    shapes = labelme_json['shapes']
    for l in shape_labels:
        shapes = list(filter(lambda shape: shape['label'] != l or shape['shape_type'] != 'point', shapes))
    labelme_json['shapes'] = shapes
    return labelme_json


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

    Parallel(n_jobs=n_jobs)(delayed(do)(fn) for fn in tqdm(input_dir.glob('*.json')))


def load_ndjson(filename: Union[str, Path], encoding=None) -> List[Dict]:
    with open(filename, encoding=encoding) as f:
        return [json.loads(line) for line in f]


def save_ndjson(ndjson: List[Any], filename: Union[str, Path], encoding=None, ensure_ascii=False):
    with open(filename, 'w', encoding=encoding) as f:
        for data in ndjson:
            f.write(json.dumps(data, ensure_ascii=ensure_ascii, separators=(',', ':')))
            f.write('\n')


def diff_ndjsons(fn1, fn2):
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


def copy_image(inname: str, outname: str):
    outdir = Path(outname)
    if not outdir.exists():
        print(outdir, ' not found')
        return
    if inname == '-':
        ndjson = [json.loads(line) for line in sys.stdin]
        indir = Path('.')
    else:
        ndjson = load_ndjson(inname)
        indir = Path(inname).parent
    for js in ndjson:
        src = indir / js['imagePath']
        # print(src)
        try:
            shutil.copy(src, outdir)
        except Exception as e:
            logger.error('Skipping error:%s', e)


def sort_kps(points):
    '''Sort keypoints by y position in ascending order

    Args:
        points (Dict[str, points]): keypoints
    '''
    kps = {k: np.array(sorted(v, key=lambda e: e[1])) for k, v in points.items()}
    return kps


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Utility functions for labelme.')
    subparsers = parser.add_subparsers()

    p_reset = subparsers.add_parser(
        'reset', help="Reset (empty flags and shapes) labeme files in the labelme directory."
    )
    p_reset.add_argument('input', help='Input directory')

    def command_reset(args):
        reset(args.input, len(os.sched_getaffinity(0)))

    p_reset.set_defaults(func=command_reset)

    p_diff = subparsers.add_parser('diff', help="Compare two ndjsons.")
    p_diff.add_argument('input1', help='Input ndjson')
    p_diff.add_argument('input2', help='Input ndjson')

    def command_diff(args):
        diff_ndjsons(args.input1, args.input2)

    p_diff.set_defaults(func=command_diff)

    p_copy = subparsers.add_parser('copy', help="Copy images in ndjson.")
    p_copy.add_argument('input', help='Input ndjson. Specify `-` to use stdin')
    p_copy.add_argument('output', help='Output directory')

    def command_copy(args):
        copy_image(args.input, args.output)

    p_copy.set_defaults(func=command_copy)

    p_shape2lm = subparsers.add_parser('shape2lm', help="Convert shape dict into labelme.")
    p_shape2lm.add_argument(
        'input', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input ndjson filename'
    )

    def command_p_shape2lm(args):
        for line in args.input:
            data = ShapeDictLine.model_validate_json(line)
            labelme_data = data.into_labelme_line()
            print(labelme_data.model_dump_json())

    p_shape2lm.set_defaults(func=command_p_shape2lm)

    p_lm2shape = subparsers.add_parser('lm2shape', help="Convert labelme into shape dict.")
    p_lm2shape.add_argument(
        'input', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input ndjson filename'
    )

    def command_p_lm2shape(args):
        for line in args.input:
            data = LabelMeLine.model_validate_json(line)
            shape_data = data.into_shape_dict_line()
            print(shape_data.model_dump_json())

    p_lm2shape.set_defaults(func=command_p_lm2shape)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        print('No command was specified.')
        parser.print_help()
        sys.exit(1)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
