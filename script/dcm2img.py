import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import boundingbox as bb
import numpy as np
import pandas as pd
import piexif
import pydicom
import tqdm
from joblib import Parallel, delayed
from piexif import helper
from PIL import Image


def normalize(x, percentile):
    minmax = np.percentile(x, [percentile, 100 - percentile])
    x = np.clip(x, minmax[0], minmax[1]).astype(np.float32)
    return np.round(255 * (x - minmax[0]) / (minmax[1] - minmax[0])).astype(
        np.uint8)


def save_as_img(indir: Path, outdir: Path, max_size: int, row, trim=True):
    dcm = pydicom.dcmread(indir / row.filename)
    img = dcm.pixel_array
    if dcm.PhotometricInterpretation == 'MONOCHROME1':
        img = img.max() - img
    if trim:
        pmin, pmax = np.percentile(img.ravel(), [1, 99])
        bg = np.maximum(img <= pmin, img > pmax)
        bg_rate_x, bg_rate_y = bg.mean(axis=0), bg.mean(axis=1)
        bg = (bg_rate_x[np.newaxis] > .9) + (bg_rate_y[:, np.newaxis] > .9)
        bg = bg > 0
        img = bb.crop(img, bb.bbox(1 - bg))
    img = normalize(img, percentile=.1)
    img = Image.fromarray(img)
    img.thumbnail((max_size, max_size))
    uc = helper.UserComment.dump(json.dumps(row.to_dict(), ensure_ascii=False),
                                 encoding='unicode')
    exif = piexif.dump({'Exif': {piexif.ExifIFD.UserComment: uc}})
    img.save(outdir / row.output, exif=exif)


def main():
    parser = argparse.ArgumentParser(
        description='DICOM to anonymized img (jpeg or png).')
    parser.add_argument(
        'metadata',
        help=
        'metadata file made with collect_dicom_meta. `filename` and `output` fields are required'
    )
    parser.add_argument('dicom', help='input dicom directory')
    parser.add_argument('output', help='Output image directory')
    parser.add_argument('--max_size',
                        help='Maximum size of the output image: %(default)s',
                        metavar='<int>',
                        type=int,
                        default=2048)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata, encoding='utf8')
    df = df.drop_duplicates(['output'])
    df.sort_values('PatientID', inplace=True)

    dicom_dir = Path(args.dicom)
    out_root = Path(args.output)
    Parallel(n_jobs=7)(
        delayed(save_as_img)(dicom_dir, out_root, args.max_size, row)
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df)))


if __name__ == '__main__':
    sys.exit(main())
