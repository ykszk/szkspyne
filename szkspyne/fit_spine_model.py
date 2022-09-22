# %%
from pathlib import Path
import mhd
import settings
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam

ANCHOR_LABELS = ['C7-TL', 'C7-TR', 'S-TL', 'S-TR']
ANCHOR_INDICES = [settings.LABEL_NAMES.index(l) for l in ANCHOR_LABELS]

data_dir = Path('../../data/annotation/images')
image_type = 'lateral_whole'
# file_id = 'OSC00020_20060809_1'
# file_id = 'OSC00029_20061220_1'
# file_id = 'OSC00408_20150617_12'
file_id = 'OSC00411_20150805_14'

heatmaps, h = mhd.read(str(data_dir / image_type / f'{file_id}.mha'))
heatmaps = heatmaps.transpose(2, 0, 1)
anchor_maps = [heatmaps[i] for i in ANCHOR_INDICES]
# %%
import point_utils

anchor_candidates = point_utils.local_center_of_masses(anchor_maps)
anchors = []
for candidates, hm in zip(anchor_candidates, anchor_maps):
    if len(candidates) == 0:
        raise NotImplementedError
    if len(candidates) == 1:
        anchors.append(candidates[0])
    else:
        probas = [hm[int(p[0]), int(p[1])] for p in candidates]
        anchors.append(candidates[np.argmax(probas)])

# %%
if True:  #anchors[0][0] <= anchors[1][0]:
    top_anchor = anchors[0]
else:
    top_anchor = anchors[1]
if True:  #anchors[2][0] >= anchors[3][0]:
    bottom_anchor = anchors[2]
else:
    bottom_anchor = anchors[3]


class TorchPCA():

    def __init__(self, filename):
        pca = np.load(filename)
        self.pca = pca
        if pca['point_format'] != 'xy':
            raise NotImplementedError
        self.mean = np.flip(pca['mean'].reshape(-1, 2), axis=1).ravel()
        components = pca['components']
        components = np.flip(components.reshape(components.shape[0], -1, 2),
                             axis=-1).reshape(components.shape)
        self.components = components
        self.scaled_components = components * np.sqrt(
            pca['explained_variance'])[:, np.newaxis]
        self.t_scaled_components = torch.from_numpy(
            self.scaled_components.astype(np.float32))
        point_counts = pca['point_counts']
        self.split = [0] + np.cumsum(2 * point_counts).tolist()

    def warp(self, coefs):
        '''
        warp model in the reference space
        '''
        v = torch.from_numpy(self.mean.astype(np.float32))
        for i, c in enumerate(coefs):
            v += c * self.t_scaled_components[i]
        return v

    def unravel(self, vec):
        '''
        unravel 1d vec into list of 1d vecs
        '''
        assert len(vec) == len(self.mean)
        unraveled = []
        for i in range(1, len(self.split)):
            start, end = self.split[i - 1], self.split[i]
            unraveled.append(vec[start:end].reshape(-1, 2))
        return unraveled


pca = TorchPCA('./pca_lateral_whole.npz')


class PatientPCA():

    def __init__(self, pca, heatmap, top_anchor, bottom_anchor):
        self.pca = pca
        self.heatmap = heatmap
        self.t_heatmap = torch.from_numpy(
            (heatmap / heatmap.max()).astype(np.float32))
        self.t_heatmap.requires_grad = True
        self.top_anchor = top_anchor
        self.t_top_anchor = torch.from_numpy(
            np.array(top_anchor).astype(np.float32))
        self.bottom_anchor = bottom_anchor
        self.pt_height = bottom_anchor[0] - top_anchor[0]
        self.scale = self.pt_height / pca.pca['ref_height']

    def ref2pt(self, vec):
        '''
        Map points from reference space to patient space
        '''
        vec2d = vec.reshape(-1, 2)
        return (self.scale * vec2d + self.t_top_anchor).ravel()

    def warp(self, coefs):
        '''
        warp model in the patient space
        '''
        return self.ref2pt(self.pca.warp(coefs))


import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def lerp2d(arr, idx):
    '''
    Args:
        arr: 2d torch tensor
        idx: [[y, x], ...]
    '''
    ia = torch.floor(idx).long()
    ib = ia + 1
    a, b = arr[ia[:, 0], ia[:, 1]], arr[ia[:, 0], ib[:, 1]]
    c, d = arr[ib[:, 0], ia[:, 1]], arr[ib[:, 0], ib[:, 1]]
    wx = idx[:, 1] - ia[:, 1]
    wy = idx[:, 0] - ia[:, 0]
    ab, cd = torch.lerp(a, b, wx), torch.lerp(c, d, wx)
    return torch.lerp(ab, cd, wy)


n_comps = 60

coefs = Variable(torch.FloatTensor(np.zeros((n_comps), dtype=np.float32)),
                 requires_grad=True)

for blur_sigma in [10, 5, 1]:
    if blur_sigma == 0:
        blurred_hm = heatmaps[:4]
    else:
        blurred_hm = np.array(
            [gaussian_filter(hm, blur_sigma) for hm in heatmaps[:4]])
    pt_pca = PatientPCA(pca, blurred_hm, top_anchor, bottom_anchor)
    optimizer = Adam([coefs], lr=.05, maximize=True)

    history = []
    for i in range(100):
        warped = pca.unravel(pt_pca.warp(coefs))
        probas = []
        for ch, pts in zip(pt_pca.t_heatmap, warped):
            probas.append(lerp2d(ch, pts))
        loss = torch.log(torch.concat(probas)).mean()
        history.append(loss.detach().item())
        # print(i, loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    plt.plot(history, label=f'{blur_sigma}')
plt.legend()
# %%
import matplotlib.pyplot as plt

mip = heatmaps.max(axis=0)
warped = pca.unravel(pt_pca.ref2pt(pca.warp(coefs.detach().numpy())))
plt.figure(figsize=(10, 10))
plt.imshow(mip, 'gray')
for m in warped:
    plt.scatter(m[:, 1], m[:, 0], s=10)

# %%
from PIL import Image

bg = Image.open(str(data_dir / image_type / f'{file_id}.jpg'))
original_shape = bg.size[::-1]
bg.thumbnail((2000, 768))
plt.figure(figsize=(10, 10))
plt.imshow(np.array(bg), 'gray')
for m in warped:
    plt.scatter(m[:, 1], m[:, 0], s=10)
# %%
from szkspyne import labelme_utils
import json

s = original_shape[0] / 768
scaled_warped = [(s * w.numpy()).tolist() for w in warped]
json_data = labelme_utils.kps2labelme(scaled_warped, 'tmp.jpg', original_shape,
                                      ['TL', 'TR', 'BL', 'BR'])
with open('tmp/tmp.json', 'w') as f:
    json.dump(json_data, f)
