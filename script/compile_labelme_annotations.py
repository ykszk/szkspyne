# %%
from pathlib import Path
import json
from logzero import logger
import numpy as np

from szkspyne import labelme_utils

data_root = Path('../../data')
indir = data_root / 'annotation/annotated/'
imgdir = data_root / 'jpg_half'
sorted_fns = sorted(indir.glob('**/valid*.ndjson'))
logger.info('file list done')
logger.info(sorted_fns)
# %%
ADD_C7S = True
logger.info('start')
dataset = {}
for fn in sorted_fns:
    with open(str(fn)) as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        filename = '/'.join(
            Path(data['filepath']).with_suffix('.jpg').name.split('_'))
        filepath = (imgdir / filename).relative_to(data_root)
        filepath = str(filepath).replace('\\', '/')
        shape_data = labelme_utils.conv2shape_dict(data)
        if ADD_C7S:
            d_points = shape_data['point']
            tl = np.array(sorted(d_points['TL'], key=lambda e: e[0][1]))[:, 0]
            tr = np.array(sorted(d_points['TR'], key=lambda e: e[0][1]))[:, 0]
            d_points['C7-TL'] = [[tl[0].tolist()]]
            d_points['C7-TR'] = [[tr[0].tolist()]]
            d_points['S-TL'] = [[tl[-1].tolist()]]
            d_points['S-TR'] = [[tr[-1].tolist()]]
        shape_data['filepath'] = filepath
        shape_data['flags'] = [k for k, v in data['flags'].items() if v]
        dataset[filepath] = shape_data
logger.info('done processing')
print(len(dataset))
# %%
outname = 'manual_annotation.ndjson'
with open(outname, 'w') as f:
    for data in dataset.values():
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')
# %%
