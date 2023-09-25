import sys
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Detectron2 prediction to labelme.')
    parser.add_argument('input', help='Input ndjson filename')
    parser.add_argument('metadata', help='Metadata json filename')
    parser.add_argument('--ext', help='Image file extension. default: %(default)s', default='.jpg')

    args = parser.parse_args()

    ndjson_filename = args.input
    with open(ndjson_filename) as f:
        predictions = [json.loads(line) for line in f]

    metadata_filename = args.metadata
    with open(metadata_filename) as f:
        metadata = json.load(f)

    ext = args.ext
    VERSION = '5.0.1'
    labels = metadata['thing_classes']
    for pred in predictions:
        instances = pred['instances']
        height, width = instances['image_size']
        lmdata = dict(
            version=VERSION,
            flags={},
            shapes=[],
            imagePath=str(Path(pred['filename']).with_suffix(ext)),
            imageData=None,
            imageHeight=height,
            imageWidth=width,
            filename=pred['filename'],
        )
        shapes = []
        for label_id, box in zip(instances['pred_classes'], instances['pred_boxes']):
            label = labels[label_id]
            shape = dict(label=label, points=[box[:2], box[2:]], group_id=None, shape_type='rectangle', flags={})
            shapes.append(shape)
        lmdata['shapes'] = shapes
        print(json.dumps(lmdata))
        # break


if __name__ == '__main__':
    sys.exit(main())
