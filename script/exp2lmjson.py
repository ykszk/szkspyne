#%%
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='From ndjson for exp to ndjson of labelme.')
parser.add_argument('input',
                    help='Input ndjson filename. "-" to use stdin',
                    metavar='<input>')
parser.add_argument('output', help='Output directory', metavar='<output>')
args = parser.parse_args()
outdir = Path(args.output)

if args.input == '-':
    exp_data = [json.loads(l) for l in sys.stdin]
else:
    with open(args.input) as f:
        exp_data = [json.loads(l) for l in f]

if outdir.suffix=='.ndjson':
    # empty the file
    with open(outdir, 'w'):
        pass

for old in tqdm(exp_data):
    image_type = old.pop('image_type')
    flags = old.pop('flags')
    for l in ['C7-TL', 'C7-TR', 'S-TL', 'S-TR']:
        old['point'].pop(l, None)
    new_shapes = []
    for shape in ['point', 'rectangle', 'circle']:
        if shape not in old:
            continue
        for k, v in old[shape].items():
            for pts in v:
                new_shapes.append({
                    'label': k,
                    'points': pts,
                    'group_id': None,
                    'shape_type': shape,
                    'flags': {}
                })
    filepath = Path(old['filepath'])
    if filepath.parts[0] in ['SSS', 'CLO']:
        stem = filepath.stem
    else:
        stem = '_'.join(filepath.with_suffix('').parts[1:])
    new = {
        "version": "4.5.7",
        'flags': {f: True
                  for f in flags},
        'shapes': new_shapes,
        'imagePath': stem + '.jpg',
        'imageData': None,
        'imageHeight': old['imageHeight'],
        'imageWidth': old['imageWidth']
    }

    if outdir.suffix=='.ndjson':
        new['filename'] = stem
        with open(outdir, 'a') as f:
            f.write(json.dumps(new))
            f.write('\n')
    else:
        filename = (outdir / stem).with_suffix('.json')
        with open(filename, 'w') as f:
            json.dump(new, f, indent=2, ensure_ascii=False)
