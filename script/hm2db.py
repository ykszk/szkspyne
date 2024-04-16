import argparse
import sqlite3
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from szkmipy import mhd
from tqdm import tqdm

from szkspyne import labelme_utils


def main():
    parser = argparse.ArgumentParser(description='Convert heatmap to png images in sqlite3 database.')
    parser.add_argument('heatmap', help='heatmap mhd filename')
    parser.add_argument('ndjson', help='ndjson with filename keys')
    parser.add_argument('output', help='output sqlite3 database name')
    parser.add_argument(
        '--skip', help='skip saving heatmap if the image already exists in the db.', action='store_true'
    )

    args = parser.parse_args()

    heatmaps, h = mhd.read(args.heatmap)
    ndjson = labelme_utils.load_ndjson(args.ndjson)
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

    for jobj, hm in tqdm(zip(ndjson, heatmaps), total=len(heatmaps)):
        filename = str(Path(jobj['filename']).with_suffix('.png'))
        if args.skip:
            cursor.execute("SELECT 1 FROM image WHERE name=?", (filename,))
            if cursor.fetchone() is not None:
                continue
        width = int(round(jobj['content']['imageWidth'] * hm.shape[0] / jobj['content']['imageHeight']))
        hm = hm[:, :width]
        hm = hm / hm.max()
        r = hm[..., :2].max(axis=-1)  # TL and TR
        g = hm[..., 2:4].max(axis=-1)  # BL and BR
        b = hm[..., 4:].max(axis=-1)  # others
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
