#%%
import os
from pathlib import Path

from szkspyne import exp_utils, settings, transforms


LABEL_NAMES = settings.LABEL_NAMES
NUM_WORKERS = 4
height = 768
data_root = ''
df = exp_utils.ndjson2csv('../script/manual_annotation.ndjson', set())
data_root = Path('../../data')
N = int(os.environ.get('N_SAMPLES', 3))
xs, kps = exp_utils.load_dataset(df.sample(n=N),
                                 height,
                                 data_root,
                                 LABEL_NAMES,
                                 NUM_WORKERS,
                                 256,
                                 clahe=False)

# %%
import matplotlib.pyplot as plt
import numpy as np


N_AUG = int(os.environ.get('N_AUG', 5))
for i, (x, kp) in enumerate(zip(xs, kps)):
    print(i)
    for _ in range(N_AUG):
        new_img, new_kp = transforms.random_transform(x, kp,
                                                      settings.FLIP_PAIRS)
        plt.imshow(new_img[0], 'gray')
        for p in new_kp:
            if p:
                p = np.array(p)
                plt.scatter(p[:, 1], p[:, 0], s=2)
        plt.show()
