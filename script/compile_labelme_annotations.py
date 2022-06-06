#%%
from pathlib import Path
import json
import pandas as pd
from collections import defaultdict
from logzero import logger
from typing import Dict, Any

data_root = Path('../../data')
indir = data_root / 'annotation/annotated'
imgdir = data_root / 'jpg_half'
sorted_fns = sorted(indir.glob('**/*.ndjson'))
logger.info('file list done')
logger.info(len(sorted_fns))
#%%
logger.info('start')
dataset = []
label_names = ['TL', 'TR', 'BL', 'BR']
# def conv_labelme(data: dict):
for fn in sorted_fns:
    if fn.name.startswith('invalid_'):
        print('skip', fn)
        continue
    with open(str(fn)) as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        if data['flags'].get('側面右向き', False):
            print('右向き')
            raise NotImplemented
        points = [s for s in data['shapes'] if s['shape_type'] == 'point']
        point_dict = defaultdict(list)
        filename = '/'.join(
            Path(data['filepath']).with_suffix('.jpg').name.split('_'))
        filepath = (imgdir / filename).relative_to(data_root)
        filepath = str(filepath).replace('\\', '/')
        for p in points:
            point_dict[p['label']].append(p['points'][0])
        if False:
            point_labels = set([e for e in point_dict.keys()])
            if set(label_names) != point_labels:
                print('Ignoring invalid keypoints', filepath, point_labels)
                continue
        row: Dict[str, Any] = {'filepath': filepath}
        for l in label_names:
            ps = point_dict[l]
            x, y = list(zip(*ps))
            row[l + ' x'] = x
            row[l + ' y'] = y

        dataset.append(row)
#    break
logger.info('done processing')
df = pd.DataFrame(dataset)
print(df.shape)
outname = 'manual_annotation.csv'
df.to_csv(outname, index=False)
df.head()