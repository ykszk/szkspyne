#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data_dir = Path('../../data')
label_names = ['TL', 'TR', 'BL', 'BR']
#%%
from PIL import Image


for dataset_name in ['spineweb', 'verse']:
    df = pd.read_csv(dataset_name + '.csv')
    for _, row in df.iterrows():
        img = Image.open(data_dir / row.filepath)
        plt.imshow(img, 'gray')
        for ln in label_names:
            xs, ys = row.get(ln + ' x'), row.get(ln + ' y')
            xs, ys = eval(xs), eval(ys)
            plt.scatter(xs, ys, s=2)
        plt.axis('off')
        plt.gcf().savefig(dataset_name + '.svg',
                          transparent=True,
                          bbox_inches='tight')
        plt.close()
        break
# %%
