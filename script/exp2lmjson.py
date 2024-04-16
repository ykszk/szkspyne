# %%
import argparse
import json
import sys
from pathlib import Path
from szkspyne import labelme_utils
from tqdm import tqdm

parser = argparse.ArgumentParser(description='From ndjson for exp to ndjson of labelme.')
parser.add_argument('input', help='Input ndjson filename. "-" to use stdin', metavar='<input>')
parser.add_argument('output', help='Output directory', metavar='<output>')
args = parser.parse_args()
outdir = Path(args.output)

if args.input == '-':
    exp_data = [labelme_utils.ShapeDictLine.model_validate_json(l) for l in sys.stdin]
else:
    with open(args.input) as f:
        exp_data = [labelme_utils.ShapeDictLine.model_validate_json(l) for l in f]

if outdir.suffix == '.ndjson':
    # empty the file
    with open(outdir, 'w'):
        pass

for data_line in tqdm(exp_data):
    old = data_line.content
    flags = old.flags
    for l in ['C7-TL', 'C7-TR', 'S-TL', 'S-TR']:
        old.shapes['point'].pop(l, None)
    new_shapes = []
    for shape in ['point', 'rectangle', 'circle']:
        if shape not in old.shapes:
            continue
        for k, v in old.shapes[shape].items():
            for pts in v:
                new_shapes.append({'label': k, 'points': pts, 'group_id': None, 'shape_type': shape, 'flags': {}})
    image_path = Path(data_line.content.imagePath)
    new_dict = {
        "version": "4.5.7",
        'flags': {f: True for f in flags},
        'shapes': new_shapes,
        'imagePath': image_path.name,
        'imageData': None,
        'imageHeight': old.imageHeight,
        'imageWidth': old.imageWidth,
    }
    new = labelme_utils.LabelMe.model_validate(new_dict)


    if outdir.suffix == '.ndjson':
        new_line = new.into_labelme_line(data_line.filename)
        with open(outdir, 'a') as f:
            f.write(new_line.model_dump_json())
            f.write('\n')
    else:
        with open(data_line.filename, 'w') as f:
            json.dump(new, f, indent=2, ensure_ascii=False)
