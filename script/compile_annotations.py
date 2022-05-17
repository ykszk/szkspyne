from pathlib import Path
import json
import pandas as pd
from collections import defaultdict
from logzero import logger

indir = Path('../data/agreed_annotation')

logger.info('start')
dataset = []
label_names = ['C2 IA', 'C2 IP', 'C last IA', 'C last IP']
label_names = ['TL', 'TR', 'BL', 'BR']
sorted_fns = sorted(indir.glob('**/*.json'))
logger.info('file list done')
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
    row = {'filepath': filepath}
    for l in label_names:
        ps = point_dict[l]
        if len(ps) > 1:
            print('Too many points!', filepath)
            continue
        p = ps[0]
        row[l + ' x'] = p[0]
        row[l + ' y'] = p[1]
        
    dataset.append(row)
#    break
logger.info('done processing')
df = pd.DataFrame(dataset)
df.to_csv('dataset.csv', index=False)