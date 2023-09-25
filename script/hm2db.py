from pathlib import Path
from szkspyne import labelme_utils
from szkmipy import mhd
from PIL import Image
import numpy as np
from io import BytesIO
import sqlite3
from tqdm import tqdm
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Convert heatmap to png images in sqlite3 database.')
    parser.add_argument('heatmap', help='heatmap mhd filename')
    parser.add_argument('ndjson', help='ndjson with filename keys')
    parser.add_argument('output', help='output sqlite3 database name')

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
        width = int(round(jobj['imageWidth'] * hm.shape[0] / jobj['imageHeight']))
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
        filename = jobj['filename'] + '.png'
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
