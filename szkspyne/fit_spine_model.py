# %%
from pathlib import Path
import mhd
import settings
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from logzero import logger
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from PIL import Image
from szkspyne import labelme_utils
import json

ANCHOR_LABELS = ['C7-TL', 'C7-TR', 'S-TL', 'S-TR']
ANCHOR_INDICES = [settings.LABEL_NAMES.index(l) for l in ANCHOR_LABELS]
IMG_HEIGHT = 768

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


N_ITER = 100
LR = 0.05
d_loss = {}
for name, image_type in (('L4', image_type), ('L5', image_type)):
    pca = TorchPCA(f'pca_{image_type}_{name}.npz')

    n_comps = 50

    coefs = Variable(torch.FloatTensor(np.zeros((n_comps), dtype=np.float32)),
                     requires_grad=True)

    for blur_sigma in [10, 1]:
        logger.info('sigma %f', blur_sigma)
        if blur_sigma == 0:
            blurred_hm = heatmaps[:4]
        else:
            blurred_hm = np.array(
                [gaussian_filter(hm, blur_sigma) for hm in heatmaps[:4]])
        pt_pca = PatientPCA(pca, blurred_hm, top_anchor, bottom_anchor)
        optimizer = Adam([coefs], lr=LR, maximize=True)

        history = []
        for i in range(N_ITER):
            warped = pca.unravel(pt_pca.warp(coefs))
            probas = []
            for ch, pts in zip(pt_pca.t_heatmap, warped):
                probas.append(lerp2d(ch, pts))
            loss = torch.log(torch.concat(
                probas)).mean()  # - 0.01 * torch.linalg.norm(coefs)
            history.append(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        plt.plot(history, label=f'{blur_sigma}')
    plt.legend()
    d_loss[name] = history[0]

    bg = Image.open(str(data_dir / image_type / f'{file_id}.jpg'))
    original_shape = bg.size[::-1]

    # mip = heatmaps.max(axis=0)
    mip = heatmaps[:3].transpose((1, 2, 0)).copy()
    mip[:, :, 0] = np.maximum(mip[:, :, 0], heatmaps[3])
    mip = mip[:, :int((IMG_HEIGHT / original_shape[0]) * original_shape[1])]
    mip[..., 0] = 0
    warped = pca.unravel(pt_pca.ref2pt(pca.warp(coefs.detach().numpy())))
    if True:
        plt.figure(figsize=(10, 10))
        plt.imshow(mip, 'gray')
        for m in warped:
            plt.scatter(m[:, 1], m[:, 0], s=10)
        plt.show()

    bg.thumbnail((IMG_HEIGHT, IMG_HEIGHT))
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(bg), 'gray')
    for m in warped:
        plt.scatter(m[:, 1], m[:, 0], s=10)
    plt.title(name)
    plt.show()

    s = original_shape[0] / IMG_HEIGHT
    scaled_warped = [(s * w.numpy()).tolist() for w in warped]
    json_data = labelme_utils.kps2labelme(scaled_warped, 'tmp.jpg',
                                          original_shape,
                                          ['TL', 'TR', 'BL', 'BR'])
    with open(f'tmp/{name}_{image_type}.json', 'w') as f:
        json.dump(json_data, f)
