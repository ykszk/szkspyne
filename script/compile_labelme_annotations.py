#%%
from pathlib import Path
import json
from logzero import logger
from typing import Dict, Any
import sys

sys.path.append('..')
sys.path.append('../spinex')
from spinex import labelme_utils

data_root = Path('../../data')
indir = data_root / 'annotation/annotated/'
imgdir = data_root / 'jpg_half'
sorted_fns = sorted(indir.glob('**/*.ndjson'))
logger.info('file list done')
logger.info(len(sorted_fns))
#%%
logger.info('start')
dataset = []
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
        filename = '/'.join(
            Path(data['filepath']).with_suffix('.jpg').name.split('_'))
        filepath = (imgdir / filename).relative_to(data_root)
        filepath = str(filepath).replace('\\', '/')
        shape_data = labelme_utils.conv2shape_dict(data)
        shape_data['filepath'] = filepath
        dataset.append(shape_data)
#    break
logger.info('done processing')
print(len(dataset))
#%% check for duplications
filepaths = [d['filepath'] for d in dataset]
assert len(dataset)==len(set(filepaths))
#%%
outname = 'manual_annotation.ndjson'
with open(outname, 'w') as f:
    for data in dataset:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')
# df.to_csv(outname, index=False)
# df.head()

#%%
