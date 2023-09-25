# %%
from pathlib import Path
import sys
import argparse
import json
import sys
import argparse
from szkspyne import settings

from logzero import logger

# [Use Custom Datasets — detectron2 0.6 documentation](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)

META_VERTEBRA = dict(
    thing_classes=["vertebra"], keypoint_names=['TL', 'TR', 'BL', 'BR'], keypoint_flip_map=[('TL', 'TR'), ('BL', 'BR')]
)
META_IMPLANT = dict(thing_classes=settings.IMPLANT_LABEL_NAMES, keypoint_names=['center'], keypoint_flip_map=[])


def main():
    parser = argparse.ArgumentParser(description='Detectron2 utility tools.')
    parser.add_argument('input', help='Input labelme ndjson. Specify "-" to use stdin', metavar='<input>')
    parser.add_argument('output', help='Output ndjson filename', metavar='<output>')
    parser.add_argument(
        '--mode', help='Dataset mode. default: %(default)s', choices=['vertebra', 'implant'], default='vertebra'
    )
    parser.add_argument('--metadata', help='Save metadata in json')
    parser.add_argument(
        '--image_dir', help='Image directory. default: %(default)s', default='../../data/annotation/images/all'
    )

    args = parser.parse_args()

    # lazy load modules
    import detectron2.structures
    import numpy as np
    from szkspyne import labelme_utils
    from tqdm import tqdm

    image_dir = Path(args.image_dir)

    if args.input == '-':
        lines = [line for line in sys.stdin]
    else:
        with open(args.input) as f:
            lines = [line for line in f]
    new_dataset = []
    for line in tqdm(lines):
        original = json.loads(line)

        dtron2_data = {
            'file_name': str(image_dir / original['imagePath']),
            'height': original['imageHeight'],
            'width': original['imageWidth'],
            'image_id': original['imagePath'],
            'annotations': [],
        }
        BBOX_MODE = detectron2.structures.BoxMode.XYXY_ABS
        VISIBILITY = 2  # all points are visible

        shape_dict = labelme_utils.conv2shape_dict(original)
        if args.mode == 'vertebrae':
            kps = {}
            for l in ['TL', 'TR', 'BR', 'BL']:
                kps[l] = np.array(shape_dict['point'][l])[:, 0]  # remove redundant dimension
            kps = labelme_utils.sort_kps(kps)

            # remove sacral points
            sacral_tl = kps['TL'][-1]
            sacral_tr = kps['TR'][-1]
            kps['TL'] = kps['TL'][:-1]
            kps['TR'] = kps['TR'][:-1]

            if not (len(kps['TL']) == len(kps['TR']) == len(kps['BR']) == len(kps['BL'])):
                raise RuntimeError(
                    'Invalid number of keypoints: %d, %d, %d, %d',
                    len(kps['TL']),
                    len(kps['TR']),
                    len(kps['BR']),
                    len(kps['BL']),
                )

            vert_corners = np.stack([kps['TL'], kps['TR'], kps['BL'], kps['BR']]).transpose(
                (1, 0, 2)
            )  # [vert, corner, xy]
            assert vert_corners.shape[1] == 4 and vert_corners.shape[2] == 2

            CATEGORY_ID = 0
            MARGIN = 32
            for corners in vert_corners:
                tlx, tly = corners[:, 0].min(), corners[:, 1].min()
                brx, bry = corners[:, 0].max(), corners[:, 1].max()
                keypoints = np.append(corners, np.full([corners.shape[0], 1], VISIBILITY), axis=1)
                anno = {
                    'bbox': [tlx - MARGIN, tly - MARGIN, brx + MARGIN, bry + MARGIN],
                    'bbox_mode': BBOX_MODE,
                    'category_id': CATEGORY_ID,
                    'keypoints': keypoints.flatten().tolist(),
                }
                dtron2_data['annotations'].append(anno)
        elif args.mode == 'implant':
            label2id = {label: i for i, label in enumerate(settings.IMPLANT_LABEL_NAMES)}
            rects = shape_dict.get('rectangle', {})
            for label, bboxes in rects.items():
                for bbox in bboxes:
                    bbox = np.array(bbox)
                    if bbox.size != 4:
                        logger.warning('Skipping invalid rectangle points: %s, in %s', bbox, original['imagePath'])
                        continue
                    anno = {
                        'bbox': bbox.reshape(-1).tolist(),
                        'bbox_mode': BBOX_MODE,
                        'category_id': label2id[label],
                        'keypoints': bbox.mean(axis=0).tolist() + [VISIBILITY],
                    }
                    dtron2_data['annotations'].append(anno)
        else:
            raise RuntimeError(f'Invalid mode: {args.mode}')
        new_dataset.append(dtron2_data)

    with open(args.output, 'w') as f:
        for data in new_dataset:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

    if args.metadata:
        meta = META_VERTEBRA if args.mode == 'vertebra' else META_IMPLANT
        with open(args.metadata, 'w') as f:
            json.dump(meta, f, ensure_ascii=False)

    return 0


if __name__ == '__main__':
    sys.exit(main())
