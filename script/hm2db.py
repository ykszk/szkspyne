import argparse
import sqlite3
import sys
from io import BytesIO
from pathlib import Path
import json

import numpy as np
from PIL import Image
from szkmipy import mhd
from tqdm import tqdm
import re

from szkspyne import labelme_utils
from szkspyne.utils import CompDecompGenerator

# RE_PATTERN_SLICE = r'(?P<start>-?\d+),(?P<stop>-?\d+),(?P<step>-?\d+)'
RE_PATTERN_SLICE = r'(?P<start>-?\d+)(,(?P<stop>-?\d+))?(,(?P<step>-?\d+))?'

def parse_slice(slice_str):
    """
    Parse a slice string into a tuple of (start, stop, step).
    """
    match = re.match(RE_PATTERN_SLICE, slice_str)
    if not match:
        raise ValueError(f"Invalid slice string: {slice_str}")
    
    start = int(match.group('start'))
    stop = int(match.group('stop')) if match.group('stop') else None
    step = int(match.group('step')) if match.group('step') else 1
    
    return slice(start, stop, step)

def main():
    parser = argparse.ArgumentParser(description='Convert heatmap to png images in sqlite3 database.')
    parser.add_argument('heatmap', help='heatmap mhd filename or directory with mha files', type=Path)
    parser.add_argument('ndjson', help='ndjson with filename keys or directory with json files', type=Path)
    parser.add_argument('output', help='output sqlite3 database name')
    parser.add_argument('--abspath', help='use absolute path for filename', action='store_true')
    parser.add_argument(
        '--skip', help='skip saving heatmap if the image already exists in the db.', action='store_true'
    )
    parser.add_argument('-r', '--red',help='Slice (start,stop,step) for red', type=str, default='0,2,1')
    parser.add_argument('-g', '--green', help='Slice (start,stop,step) for green', type=str, default='2,4,1')
    parser.add_argument('-b', '--blue', help='Slice (start,stop,step) for blue', type=str, default='4,,1')

    args = parser.parse_args()

    if args.ndjson.is_dir():
        ndjson = []
        for fn in sorted(args.ndjson.glob('*.json')):
            with open(fn) as f:
                jboj = json.load(f)
            ndjson.append({'filename': fn.name, 'content': jboj})
    else:
        ndjson = labelme_utils.load_ndjson(args.ndjson)

    if args.heatmap.is_dir():
        heatmaps = []
        for fn in sorted(args.heatmap.glob('*.mha')):
            arr, _h = mhd.read(fn)
            heatmaps.append(arr)
        n_heatmaps = len(heatmaps)
    else:
        # heatmaps, _h = mhd.read(args.heatmap)
        # heatmaps = [h.copy() for h in heatmaps]
        # heatmaps = CompDecompGenerator.builder(method='zstandard', level=1)(heatmaps)
        heatmaps, header = mhd.read_iterator(args.heatmap)
        n_heatmaps = header['DimSize'][-1]
    assert n_heatmaps == len(ndjson)

    conn = sqlite3.connect(args.output)
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS image (
            name TEXT PRIMARY KEY,
            binary BLOB
        )
    '''
    )
    conn.commit()

    s_red = parse_slice(args.red)
    s_green = parse_slice(args.green)
    s_blue = parse_slice(args.blue)

    for jobj, hm in tqdm(zip(ndjson, heatmaps), total=n_heatmaps):
        filename = str(Path(jobj['filename']).with_suffix('.png'))
        if not args.abspath:
            filename = Path(filename).name
        if args.skip:
            cursor.execute("SELECT 1 FROM image WHERE name=?", (filename,))
            if cursor.fetchone() is not None:
                continue
        width = int(round(jobj['content']['imageWidth'] * hm.shape[0] / jobj['content']['imageHeight']))
        hm = hm[:, :width]
        hm = hm / hm.max()
        r = hm[..., s_red].max(axis=-1)  # TL and TR
        g = hm[..., s_green].max(axis=-1)  # BL and BR
        b = hm[..., s_blue].max(axis=-1)  # others
        rgb = np.stack((r, g, b), axis=-1)
        alpha = rgb.max(axis=-1)
        rgba = np.concatenate((rgb, alpha[..., np.newaxis]), axis=-1)
        rgba = np.round(255 * rgba).astype(np.uint8)
        img = Image.fromarray(rgba)
        with BytesIO() as bio:
            img.save(bio, format='png')
            binary = bio.getvalue()
        cursor.execute(
            '''
        REPLACE INTO image (name, binary) VALUES (?, ?)
    ''',
            (filename, binary),
        )
        conn.commit()

    conn.close()


if __name__ == '__main__':
    sys.exit(main())
