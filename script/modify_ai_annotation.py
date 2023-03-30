#%%
from pathlib import Path
import json
from logzero import logger
from typing import Any

from szkspyne import labelme_utils, img_utils
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Modify AI annotation.')
parser.add_argument('image_type',
                    help='image_type',
                    choices=['ap_whole', 'lateral_whole'])
parser.add_argument('input', help='input directory')
parser.add_argument('-o',
                    '--output',
                    help='output directory. input is used by default')
args = parser.parse_args()
root_dir = Path(args.input)

AP_ONLY = set(['Shoulder', 'Clavicle', 'Pelvis', 'Iliac'])


class JsonsDirIO:

    class JsonIO:

        def __init__(self, infile: Path, outfile: Path):
            self.infile = infile
            self.outfile = outfile

        def read(self) -> Any:
            with open(self.infile) as f:
                return json.load(f)

        def write(self, obj: Any):
            with open(self.outfile, 'w') as f:
                json.dump(obj, f)

    def __init__(self, indir: Path, outdir: Path):
        if indir.is_file():
            raise RuntimeError(f'{indir} is a file.')
        if not indir.exists():
            raise RuntimeError(f'{indir} does not exist.')

        if outdir.is_file():
            raise RuntimeError(f'File {outdir} already exists.')
        if not outdir.exists():
            outdir.mkdir(parents=True)
        self.outdir = outdir
        self.fns = sorted(indir.glob('*.json'))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, i: int):
        if i >= len(self):
            raise IndexError
        return self.JsonIO(self.fns[i], self.outdir / self.fns[i].name)


class NDJsonIO:

    class LineIO:

        def __init__(self, data: str, ndjson: dict, i: int):
            self.data = data
            self.ndjson = ndjson
            self.i = i

        def read(self) -> Any:
            return self.data

        def write(self, obj: Any):
            line = json.dumps(obj, ensure_ascii=False)
            self.ndjson[self.i] = line

    def __init__(self, input: Path, output: Path):
        with open(input) as f:
            self.data = [json.loads(line) for line in f]
        self.output = output
        self.ndjson = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        with open(self.output, 'w') as f:
            for i in sorted(self.ndjson.keys()):
                f.write(self.ndjson[i])
                f.write('\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        if i >= len(self):
            raise IndexError
        return self.LineIO(self.data[i], self.ndjson, i)


input = Path(args.input)
if args.output:
    output = Path(args.output)
else:
    output = input

with JsonsDirIO(input, output) if input.is_dir() else NDJsonIO(input,
                                                               output) as ios:
    if args.image_type == 'lateral_whole':
        # convert FemoralHead points to circles based on average vertebra sizes
        logger.info('lateral')
        for rw in ios:
            data = rw.read()
            fhs = []
            shapes = data['shapes']
            shapes = list(filter(lambda e: e['label'] not in AP_ONLY, shapes))
            data['shapes'] = shapes
            for shape in data['shapes']:
                if shape['shape_type'] == 'point' and shape[
                        'label'] == 'FemoralHead':
                    fhs.append(shape)
            if len(fhs) > 0:
                shapes = labelme_utils.conv2shape_dict(data)
                # calculate median disk width
                disk_widths = []
                for pair in [('TL', 'TR'), ('BL', 'BR')]:
                    l, r = np.array(
                        shapes['point'][pair[0]]).squeeze(), np.array(
                            shapes['point'][pair[1]]).squeeze()
                    dists = cdist(l, r)
                    less_axis = 0 if len(l) <= len(r) else 1
                    disk_widths.append(dists.min(axis=less_axis))
                median_width = np.median(np.concatenate(disk_widths))
                # point -> circle
                for fh in fhs:
                    fh['shape_type'] = 'circle'
                    center = np.array(fh['points'][0])
                    arc_top = center.copy()
                    arc_top[1] -= .8 * median_width
                    points = np.stack((center, arc_top))
                    fh['points'] = points.tolist()
                rw.write(data)

    elif args.image_type == 'ap_whole':
        HEIGHT = 768
        logger.info('ap')
        for rw in ios:
            data = rw.read()
            shapes = labelme_utils.conv2shape_dict(data)
            all_points = []
            for label in ['TL', 'TR', 'BL', 'BR']:
                if label in shapes['point']:
                    all_points.append(np.array(shapes['point'][label]))
            all_points = np.concatenate(all_points)[:, 0]
            x_middle = all_points[:, 0].mean()
            for label in [
                    'Shoulder', 'Clavicle', 'Pelvis', 'Iliac', 'FemoralHead'
            ]:
                lefts, rights = [], []
                for p in shapes['point'].get(label, []):
                    if p[0][0] > x_middle:
                        rights.append(p)
                    else:
                        lefts.append(p)
                for side in [lefts, rights]:
                    if len(side) >= 2:
                        side = np.array(side)
                        del_shapes = []
                        if False:  # heatmap:  # choose most likely point using heatmap
                            raise NotImplementedError
                        else:  # choose highest (lowest in image) point
                            ys = side[:, 0, 1]
                            highest = np.argmin(ys)
                            for shape in data['shapes']:
                                if shape['shape_type'] == 'point' and shape[
                                        'label'] == label:
                                    y = shape['points'][0][1]
                                    if y > highest:
                                        del_shapes.append(shape)
                        for ds in del_shapes:
                            data['shapes'].remove(ds)
            rw.write(data)
    else:
        print('Invalid image_type', args.image_type)
