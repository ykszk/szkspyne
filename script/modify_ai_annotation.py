#%%
from pathlib import Path
import json
from logzero import logger

from szkspyne import labelme_utils, img_utils
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm
import mhd
import argparse

parser = argparse.ArgumentParser(description='Modify AI annotation.')
parser.add_argument(
    'root',
    help='root directory with ap_whole and lateral_whole directory',
    metavar='<input>')
args = parser.parse_args()
root_dir = Path(args.root)

# convert FemoralHead points to circles based on average vertebra sizes
logger.info('lateral')
indir = root_dir / 'lateral_whole'
for filename in tqdm(sorted(indir.glob('*.json'))):
    with open(filename) as f:
        data = json.load(f)
    fhs = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'point' and shape['label'] == 'FemoralHead':
            fhs.append(shape)
    if len(fhs) > 0:
        shapes = labelme_utils.conv2shape_dict(data)
        # calculate median disk width
        disk_widths = []
        for pair in [('TL', 'TR'), ('BL', 'BR')]:
            l, r = np.array(shapes['point'][pair[0]]).squeeze(), np.array(
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
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        # break
# %%
# regulate the number of points

HEIGHT = 768
logger.info('ap')
indir = root_dir / 'ap_whole'
for filename in sorted(indir.glob('*.json')):
    with open(filename) as f:
        data = json.load(f)
    fn_heatmap = filename.with_suffix('.mha')
    if fn_heatmap.exists():
        pass
        # heatmap, _ = mhd.read(str(fn_heatmap))
        # heatmap = np.transpose(heatmap, [2, 0, 1])  # (y,x,ch) to (ch,y,x)
        # heatmap = [
        #     img_utils.resize_height(HEIGHT, hm, [])[0] for hm in heatmap
        # ]
    else:
        heatmap = None

    shapes = labelme_utils.conv2shape_dict(data)
    all_points = []
    for label in ['TL', 'TR', 'BL', 'BR']:
        if label in shapes['point']:
            all_points.append(np.array(shapes['point'][label]))
    all_points = np.concatenate(all_points)[:, 0]
    x_middle = all_points[:, 0].mean()
    write_needed = False
    for label in ['Shoulder', 'Clavicle', 'Pelvis', 'Iliac', 'FemoralHead']:
        lefts, rights = [], []
        for p in shapes['point'].get(label, []):
            if p[0][0] > x_middle:
                rights.append(p)
            else:
                lefts.append(p)
        for side in [lefts, rights]:
            if len(side) >= 2:
                write_needed = True
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
    if write_needed:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    # calculate middle line
    # break
# %%
