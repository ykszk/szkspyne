#%%
from pathlib import Path
import json
import pandas as pd
from collections import defaultdict
from logzero import logger
data_root = Path('../../data')
indir = data_root / 'verse/mip'
sorted_fns = sorted(indir.glob('**/*.json'))
logger.info('file list done')
logger.info(len(sorted_fns))
#%%
logger.info('start')
dataset = []
label_names = ['TL', 'TR', 'BL', 'BR']
for fn in sorted_fns:
    with open(str(fn)) as f:
        data = json.load(f)
    points = [s for s in data['shapes'] if s['shape_type']=='point']
    point_dict = defaultdict(list)
    filepath = fn.relative_to(indir).parent / data['imagePath']
    filepath = str(filepath).replace('\\','/')
    for p in points:
        point_dict[p['label']].append(p['points'][0])
    point_labels = set([e for e in point_dict.keys()])
    if set(label_names) != point_labels:
        print('Ignoring invalid keypoints', filepath)
        continue
    row = {'filepath': (indir / filepath).relative_to(data_root)}
    for l in label_names:
        ps = point_dict[l]
        x, y = list(zip(*ps))
        row[l + ' x'] = x
        row[l + ' y'] = y
        
    dataset.append(row)
#    break
logger.info('done processing')
df = pd.DataFrame(dataset)
df.to_csv('verse.csv', index=False)
df.head()
# %%
df_read = pd.read_csv('verse.csv')
df_read.head()
# %%
