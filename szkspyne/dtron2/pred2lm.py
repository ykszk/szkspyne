import sys
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Detectron2 prediction to labelme.')
    parser.add_argument('input', help='Input ndjson filename or directory')
    parser.add_argument('metadata', help='Metadata json filename')
    parser.add_argument('--output', help='Output filename')
    parser.add_argument('--ext', help='Image file extension. default: %(default)s', default='.jpg')

    args = parser.parse_args()

    if Path(args.input).is_dir():
        input_dir = Path(args.input)
        predictions = []
        for filename in sorted(input_dir.glob('*.json')):
            with open(filename) as f:
                pred = json.load(f)
                pred['filename'] = filename.name
                predictions.append(pred)
    else:
        ndjson_filename = args.input
        with open(ndjson_filename) as f:
            predictions = [json.loads(line) for line in f]

    metadata_filename = args.metadata
    with open(metadata_filename) as f:
        metadata = json.load(f)

    lines = []

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
        )
        shapes = []
        for label_id, box in zip(instances['pred_classes'], instances['pred_boxes']):
            label = labels[label_id]
            shape = dict(label=label, points=[box[:2], box[2:]], group_id=None, shape_type='rectangle', flags={})
            shapes.append(shape)
        lmdata['shapes'] = shapes
        lmdata_line = {'content': lmdata, 'filename': pred['filename']}
        if args.output:
            lines.append(lmdata_line)
        else:
            print(json.dumps(lmdata_line, ensure_ascii=False, separators=(',', ':')))
        # break

    if args.output:
        output_filename = args.output
        if not output_filename.endswith('.ndjson'):
            output_filename += '.ndjson'
        with open(output_filename, 'w') as f:
            for line in lines:
                json.dump(line, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')
        print(f'Output: {output_filename}')


if __name__ == '__main__':
    sys.exit(main())
