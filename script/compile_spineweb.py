#%%
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.io

sw_dir = Path('../../data/spineweb')
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
    break
df = pd.DataFrame(rows)
df.to_csv('spineweb.csv', index=False)
df.head()
# %%
