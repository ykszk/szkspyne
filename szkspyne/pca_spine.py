#%%
import sys

sys.path.append('../')
import json
import numpy as np

# filename = 'manual_annotation.ndjson'
from szkspyne import labelme_utils

filename = '../script/manual_annotation.ndjson'
image_type = 'lateral_whole'
with open(filename) as f:
    # data = [labelme_utils.conv2shape_dict(json.loads(line)) for line in f]
    data = [json.loads(line) for line in f]

LABELS = ['TL', 'TR', 'BL', 'BR']
POINT_COUNTS = [19, 19, 18, 18]

valid_data = []
for d in data:
    if d['image_type'] != image_type:
        continue
    ns = {k: len(v) for k, v in d['point'].items()}
    np_pts = {}
    for label, pts in d['point'].items():
        pts.sort(key=lambda v: v[0][1])
        np_pts[label] = np.array(pts)[:, 0]
    d['point'] = np_pts
    valid = True
    for l, n in zip(
            LABELS,
            POINT_COUNTS):  #[('TL', 19), ('TR',19), ('BL',18), ('BR',18)]:
        if ns[l] != n:
            valid = False
            break
    if valid:
        valid_data.append(d)
print(len(data), len(valid_data))
# %% sort points
l_c2s = []
for vd in valid_data:
    tl = vd['point']['TL'][0, 1]
    bl = vd['point']['BL'][-1, 1]
    l_c2s.append(bl - tl)
median_data = valid_data[np.argsort(l_c2s)[len(l_c2s) // 2]]
median_height = np.median(l_c2s)
ref_btm = median_data['point']['TL'][0].copy()
# ref_btm = median_data['point']['BL'][-1][0][0]
# %%
for vd in valid_data:
    tl = vd['point']['TL'][0]
    bl = vd['point']['BL'][-1]
    height = bl[1] - tl[1]
    # print(bl, ref_btm)
    # t = bl - ref_btm
    t = tl
    # print(t)
    scale = median_height / height
    for l in ['TL', 'TR', 'BL', 'BR']:
        vd['point'][l] = scale * (vd['point'][l] - t)
# %%
import matplotlib.pyplot as plt
for vd in valid_data[:10]:
    pts = vd['point']['TL']
    plt.scatter(pts[:, 0], pts[:, 1])
# %%
vecs = []
for vd in valid_data:
    pts = []
    for l in ['TL', 'TR', 'BL', 'BR']:
        pts.append(vd['point'][l])
    pts = np.concatenate(pts).ravel()
    vecs.append(pts)
vecs = np.stack(vecs)
vecs.shape

# %%
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(vecs)
np.savez(f'pca_{image_type}.npz',
         ref_height=median_height,
         mean=pca.mean_,
         components=pca.components_,
         explained_variance=pca.explained_variance_,
         explained_variance_ratio_=pca.explained_variance_ratio_,
         point_format='xy',
         labels=LABELS,
         point_counts=POINT_COUNTS)
# %%
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumsum[:20])
#%%
m = pca.mean_


def scat(p, label=None):
    p = p.reshape(-1, 2)
    return plt.scatter(p[:, 0], p[:, 1], label=label, c='blue')


scat(m)
#%%
scat(vecs.mean(axis=0))
#%%
import matplotlib.animation as animation

SAVE_GIF = False
if SAVE_GIF:
    fig = plt.figure()
    N = 2
    figs = []
    step = .2
    SIGMA_MAX = 2
    for i in range(N):
        for p in np.arange(-SIGMA_MAX, SIGMA_MAX, step).tolist() + np.arange(
                SIGMA_MAX, -SIGMA_MAX, -step).tolist():
            v = m.copy()
            v += p * pca.components_[i] * np.sqrt(pca.explained_variance_[i])
            f = scat(v)
            plt.title(f'{i}')
            figs.append([f])
    anim = animation.ArtistAnimation(fig, figs, interval=300, repeat_delay=0)
    writer = animation.PillowWriter(fps=10)
    anim.save('pca.gif', writer=writer)
    plt.close()
