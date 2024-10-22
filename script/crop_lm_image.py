from szkspyne import labelme_utils
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Tuple
from szkspyne.labelme_utils import LabelMe
import argparse
import sys
from logzero import logger


def crop_labelme_data(
    labelme_data: LabelMe, img: np.ndarray, roi_labels, roi_margin: float
) -> Tuple[LabelMe, np.ndarray]:
    shape_data = labelme_data.into_shape_dict()
    roi_points = []
    for label in roi_labels:
        points = np.array(shape_data.shapes['point'][label])
        points = points[:, 0]
        roi_points.append(points)
    roi_points = np.concatenate(roi_points, axis=0)

    roi_bbox = np.array(
        [roi_points[:, 0].min(), roi_points[:, 1].min(), roi_points[:, 0].max(), roi_points[:, 1].max()]
    )
    roi_bbox[:2] -= roi_margin
    roi_bbox[2:] += roi_margin
    roi_bbox = roi_bbox.clip(0, [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    roi_bbox = roi_bbox.astype(int)
    roi_img = img[roi_bbox[1] : roi_bbox[3], roi_bbox[0] : roi_bbox[2]]

    cropped_data = labelme_data.shift(-roi_bbox[0], -roi_bbox[1])
    cropped_data.imageHeight = roi_img.shape[0]
    cropped_data.imageWidth = roi_img.shape[1]

    return cropped_data, roi_img


def main():
    parser = argparse.ArgumentParser("Crop labelme image using the bounding box calculated from points.")
    parser.add_argument('ndjson', help="input labelme ndjson. Specify `-` to use stdin")
    parser.add_argument('outdir', type=Path)
    parser.add_argument('roi_labels', help="Labels", type=str, nargs='+')
    parser.add_argument('--margin', help=" default:%(default)s", type=float, default=128)
    parser.add_argument('--save_json', help='Save cropped json', action='store_true')

    args = parser.parse_args()
    outdir = args.outdir
    roi_labels = args.roi_labels
    roi_margin = args.margin

    if args.ndjson == '-':
        lines = sys.stdin.readlines()
    else:
        with open(args.ndjson) as f:
            lines = f.readlines()
    for line in lines:
        labelme_data_line = labelme_utils.LabelMeLine.model_validate_json(line)
        logger.debug("Process %s", labelme_data_line.filename)
        labelme_data = labelme_data_line.content
        shape_data = labelme_data.into_shape_dict()

        img = np.array(Image.open(shape_data.imagePath))

        cropped_data, roi_img = crop_labelme_data(labelme_data, img, roi_labels, roi_margin)
        cropped_data.imagePath = Path(shape_data.imagePath).name
        cropped_data_line = labelme_utils.LabelMeLine(content=cropped_data, filename=labelme_data_line.filename)
        print(cropped_data_line.model_dump_json())
        if args.save_json:
            with open(outdir / labelme_data_line.filename, 'w') as f:
                f.write(cropped_data.model_dump_json(indent=2))

        Image.fromarray(roi_img).save(outdir / Path(shape_data.imagePath).name)


if __name__ == '__main__':
    main()
