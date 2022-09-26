# %%
from pathlib import Path
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from logzero import logger
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import json
from PIL import Image
from tqdm import tqdm

import mhd
from szkspyne import labelme_utils
from pca_spine import PointsPCA
import point_utils
import settings

ANCHOR_LABELS = ['C7-TL', 'S-TL']
ANCHOR_INDICES = [settings.LABEL_NAMES.index(l) for l in ANCHOR_LABELS]
IMG_HEIGHT = 768


def calculate_anchors(anchor_maps):
    '''
    Args:
        anchor_maps: [ch, y, x]
    '''

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

    return anchors


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

    def calc_n_comps(self, contr_rate) -> int:
        '''
        calculate the number of components required for the specified contribution ratio
        '''
        cs = np.cumsum(self.pca['explained_variance_ratio_'])
        idx = np.nonzero(cs >= contr_rate)[0][0]
        return idx + 1

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


EPSILON = np.finfo(float).eps


class PatientPCA():

    def __init__(self, pca, heatmap, top_anchor, bottom_anchor):
        self.pca = pca
        heatmap = (heatmap / heatmap.max()).astype(np.float32)
        # heatmap = np.clip(heatmap, EPSILON, None)
        self.heatmap = heatmap
        self.t_heatmap = torch.from_numpy(heatmap)
        self.t_heatmap.requires_grad = True
        self.top_anchor = top_anchor
        self.bottom_anchor = bottom_anchor
        mat_ref2pt = PointsPCA(pca.pca['ref_height']).to_target_mat(
            top_anchor[::-1], bottom_anchor[::-1]).astype(np.float32)
        # (x, y) to (y, x)
        mat_ref2pt[0, 0], mat_ref2pt[1, 1] = mat_ref2pt[1, 1], mat_ref2pt[0, 0]
        t = mat_ref2pt[:2, 2].copy()
        mat_ref2pt[0, 2], mat_ref2pt[
            1, 2] = t[1] / mat_ref2pt[-1, -1], t[0] / mat_ref2pt[-1, -1]
        self.mat_ref2pt = torch.from_numpy(mat_ref2pt)

    def ref2pt(self, vec):
        '''
        Map points from reference space to patient space
        '''
        vec2d = vec.reshape(-1, 2)
        pt_vec2d = vec2d @ self.mat_ref2pt[:2, :2] + self.mat_ref2pt[:2, 2]
        return pt_vec2d.ravel()

    def warp(self, coefs):
        '''
        warp model in the patient space
        '''
        return self.ref2pt(self.pca.warp(coefs))

    def eval_coefs(self, coefs):
        '''
        Evaluate given coefs and return probabilities of the points
        warp -> ref2pt -> unravel -> get probabilities
        '''
        warped = self.pca.unravel(self.ref2pt(self.pca.warp(coefs)))
        probas = []
        for ch, pts in zip(self.t_heatmap, warped):
            probas.append(lerp2d(ch, pts))
        return probas


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
CONTR_RATE = .999

image_type = 'lateral_whole'
data_dir = Path('../../data/annotation/images') / image_type
outdir = data_dir  #Path('tmp')
skipped = []
for fn in tqdm(sorted((data_dir).glob('*.json'))):
    file_id = fn.stem
    # logger.info(file_id)

    heatmaps, h = mhd.read(str(data_dir / f'{file_id}.mha'))
    heatmaps = heatmaps.transpose(2, 0, 1)
    anchor_maps = [heatmaps[i] for i in ANCHOR_INDICES]
    heatmaps = heatmaps[:4]

    top_anchor, bottom_anchor = calculate_anchors(anchor_maps)

    results = []
    histories = []
    for name in ('L4', 'L5'):
        pca = TorchPCA(f'pca_{image_type}_{name}.npz')

        n_comps = pca.calc_n_comps(CONTR_RATE)

        coefs = Variable(torch.FloatTensor(
            np.zeros((n_comps), dtype=np.float32)),
                         requires_grad=True)
        try:
            for blur_sigma in [10, 1]:
                if blur_sigma == 0:
                    blurred_hm = heatmaps
                else:
                    blurred_hm = np.array(
                        [gaussian_filter(hm, blur_sigma) for hm in heatmaps])
                pt_pca = PatientPCA(pca, blurred_hm, top_anchor, bottom_anchor)
                optimizer = Adam([coefs], lr=LR)

                history = []
                for i in range(N_ITER):
                    probas = pt_pca.eval_coefs(coefs)
                    loss = -torch.log(torch.concat(
                        probas)).mean() + 0.01 * torch.linalg.norm(coefs)
                    history.append(loss.detach().item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                histories.append((name, blur_sigma, history))
            coefs = coefs.detach().numpy()
        except IndexError:  # updated coefs become really big when warped points completely miss the heatmaps
            continue
        results.append((name, history[0], coefs))

    if len(results) == 0:
        skipped.append(file_id)
        continue
    with open(str(data_dir / f'{file_id}.json')) as f:
        original_json = json.load(f)
    result_json = original_json.copy()
    results.sort(key=lambda e: e[1])  # sort results by loss
    best_name, best_loss, best_coefs = results[0]
    pca = TorchPCA(f'pca_{image_type}_{best_name}.npz')
    pt_pca = PatientPCA(pca, heatmaps, top_anchor,
                        bottom_anchor)  # TODO: refactor
    warped = pca.unravel(pt_pca.ref2pt(pca.warp(best_coefs)))
    s = original_json['imageHeight'] / IMG_HEIGHT
    scaled_warped = [(s * w.numpy()).tolist() for w in warped]

    original_shapes = result_json['shapes']
    for l in ['TL', 'TR', 'BL', 'BR', 'C7-TL', 'C7-TR', 'S-TL', 'S-TR']:
        original_shapes = list(
            filter(
                lambda shape: shape['label'] != l or shape['shape_type'] !=
                'point', original_shapes))
    result_shapes = labelme_utils.kps2shapes(scaled_warped,
                                             ['TL', 'TR', 'BL', 'BR'])
    result_shapes.extend(original_shapes)
    result_json['shapes'] = result_shapes
    result_json['flags'] = {best_name: True}
    with open(outdir / f'{file_id}.json', 'w') as f:
        json.dump(result_json, f, indent=2)
with open('skipped.txt', 'w') as f:
    f.write('\n'.join(skipped))