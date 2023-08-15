#%%
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


data_dir = Path('../../data')
sw_dir = data_dir / 'spineweb'
label_names = ['TL', 'TR', 'BL', 'BR']
#%%
rows = []
for subd in ['test', 'training']:
    indir = sw_dir / 'labels' / subd
    df_filenames = pd.read_csv(indir / 'filenames.csv', names=['filename'])
    # df_lm = pd.read_csv(indir / 'landmarks.csv')
    # df_data = pd.concat((df_filenames, df_lm), axis=1)
    for fn in df_filenames['filename']:
        mat = scipy.io.loadmat(indir / (fn + '.mat'))
        points = mat['p2'].reshape((-1, len(label_names), 2))
        points_per_type = np.transpose(points, (1, 0, 2))
        row = {'filepath': f'spineweb/data/{subd}/' + fn}
        for ln, ps in zip(label_names, points_per_type):
            row[ln + ' x'] = ps[:, 0].tolist()
            row[ln + ' y'] = ps[:, 1].tolist()
        rows.append(row)
    # break
df = pd.DataFrame(rows)
df.to_csv('spineweb.csv', index=False)
df.head()
# %% labelme
import json
import sys

import tqdm


sys.path.append('..')
sys.path.append('../szkspyne')
import shutil

from szkspyne import labelme_utils


outdir = sw_dir / 'labelme'
for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    filepath = Path(row.filepath)
    dst = outdir / filepath.name
    shutil.copyfile(data_dir / filepath, dst)
    kps = []
    for label in label_names:
        kps.append(np.array((row[label + ' y'], row[label + ' x'])).T.tolist())
    lm_dict = labelme_utils.kps2labelme(kps, dst.name, [512, 512], label_names)
    with open(dst.with_suffix('.json'), 'w') as f:
        json.dump(lm_dict, f)
