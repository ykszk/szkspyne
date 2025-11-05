import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import piexif
import pydicom
from logzero import logger
from piexif import helper
from PIL import Image
from scipy import ndimage
from szkmipy import boundingbox as bb
from szkmipy import mhd


def trim(arr, percentile=10):
    orig_shape = np.array(arr.shape)
    bmin, bmax = np.zeros(arr.ndim, dtype=np.int64), np.zeros(arr.ndim, dtype=np.int64)
    # TODO: optimize
    for filt in [lambda x: x, partial(ndimage.sobel, axis=0), partial(ndimage.sobel, axis=1), ndimage.laplace] * 2:
        filt_img = filt(arr)
        min_max = np.percentile(filt_img, [percentile, 100 - percentile])
        mask = np.logical_and(min_max[0] < filt_img, filt_img < min_max[1])
        bbox = bb.bbox(mask)
        bmin = bmin + bbox[0]
        bmax = bmax + (np.array(arr.shape) - bbox[1])
        arr = bb.crop(arr, bbox, margin=0)
    return arr, (bmin, orig_shape - bmax)


def normalize(x, percentile):
    height, width = x.shape
    mh, mw = height // 10, width // 10

    if mh < 1 or mw < 1:
        tmp = x
    else:
        tmp = x[mh:-mh, mw:-mw]
    minmax = np.percentile(tmp, [percentile, 100 - percentile])
    x = np.clip(x, minmax[0], minmax[1]).astype(np.float32)
    return np.round(255 * (x - minmax[0]) / (minmax[1] - minmax[0])).astype(np.uint8), minmax.tolist()


def single_io(inname: str, outname: str, options):
    logger.info(inname)
    dcm = pydicom.dcmread(inname)
    arr = dcm.pixel_array
    if dcm.PhotometricInterpretation == 'MONOCHROME1':
        arr = arr.max() - arr
    d_exif = {'original_size': list(arr.shape)}

    if options.trim:
        arr, bbox = trim(arr)
    else:
        bbox = None

    if outname.endswith('.mha'):
        mhd.write(outname, arr)
    else:
        arr, minmax = normalize(arr, 5)
        d_exif['window_minmax'] = minmax

        img = Image.fromarray(arr)
        # arr = arr[::2, ::2]
        img.thumbnail((options.size, options.size))
        if bbox is not None:
            d_exif['crop_params'] = np.stack(bbox).ravel().tolist()

        uc = helper.UserComment.dump(json.dumps(d_exif, ensure_ascii=False), encoding='unicode')
        exif = piexif.dump({'Exif': {piexif.ExifIFD.UserComment: uc}})
        img.save(outname, exif=exif)


def main():
    parser = argparse.ArgumentParser(description='Convert (, trim, and resize) dicom file to another format.')
    parser.add_argument('input', help='Input filename')
    parser.add_argument('output', help='Output filename')
    parser.add_argument('--ext', help='Output file extension. default: %(default)s', default='.jpg')
    parser.add_argument('--size', help='Maximum image width/height. default: %(default)s', default=2048, type=int)
    parser.add_argument('--trim', help='Trim margins', action='store_true')

    args = parser.parse_args()

    if Path(args.input).is_dir():
        outdir = Path(args.output)
        if not outdir.exists():
            print(f'Output {outdir} directory not found.')
        if not outdir.is_dir():
            print('Output needs to be an directory for directory input')
            return 1
        for inname in sorted(Path(args.input).glob('*.dcm')):
            outname = outdir / inname.with_suffix(args.ext).name
            single_io(str(inname), str(outname), args)
    else:
        single_io(args.input, args.output, args)


if __name__ == '__main__':
    sys.exit(main())
