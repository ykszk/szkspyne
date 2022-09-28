# %%
from pathlib import Path
import numpy as np
import torch
from torch.optim import Adam
from logzero import logger
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import json
from PIL import Image
from tqdm import tqdm
import pandas as pd

import mhd
from szkspyne import labelme_utils
from pca_spine import PointsPCA
import point_utils
import settings
from torch_utils import lerp2d
from torch import nn

ANCHOR_LABELS = ['C7-TL', 'S-TL']
ANCHOR_INDICES = [settings.LABEL_NAMES.index(l) for l in ANCHOR_LABELS]
IMG_HEIGHT = 768
EPSILON = np.finfo(float).eps
MAX_ITER = 200
PATIENCE = 20
LR = .1
CONTR_RATE = .999
SAVE_HISTORY = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# The operator 'aten::index.Tensor' is not current implemented for the MPS device.
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

logger.info('Device:%s', device)


# %%
class TorchPCA(nn.Module):

    def __init__(self, filename):
        super().__init__()
        pca = np.load(filename)
        self.pca = pca
        if pca['point_format'] != 'xy':
            raise NotImplementedError
        self.mean = nn.parameter.Parameter(torch.from_numpy(
            np.flip(pca['mean'].reshape(-1, 2),
                    axis=1).ravel().astype(np.float32)),
                                           requires_grad=False)
        components = pca['components']
        components = np.flip(components.reshape(components.shape[0], -1, 2),
                             axis=-1).reshape(components.shape)
        scaled_components = components * np.sqrt(
            pca['explained_variance'])[:, np.newaxis]
        self.scaled_components = nn.parameter.Parameter(torch.from_numpy(
            scaled_components.astype(np.float32)),
                                                        requires_grad=False)
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
        return self.mean + torch.sum(
            torch.unsqueeze(coefs, 1) * self.scaled_components[:len(coefs)],
            dim=0)

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


class PointAlignModel(nn.Module):

    def __init__(self, pca: TorchPCA, n_comps: int, top_anchor, bottom_anchor):
        super().__init__()
        self.coefs = nn.parameter.Parameter(
            torch.zeros([n_comps], dtype=torch.float32, requires_grad=True))
        self.pca = pca
        self.top_anchor = top_anchor
        self.bottom_anchor = bottom_anchor
        mat_ref2pt = PointsPCA(pca.pca['ref_height']).to_target_mat(
            top_anchor[::-1], bottom_anchor[::-1]).astype(np.float32)
        # convert (x, y) to (y, x)
        mat_ref2pt[0, 0], mat_ref2pt[1, 1] = mat_ref2pt[1, 1], mat_ref2pt[0, 0]
        t = mat_ref2pt[:2, 2].copy()
        mat_ref2pt[0, 2], mat_ref2pt[
            1, 2] = t[1] / mat_ref2pt[-1, -1], t[0] / mat_ref2pt[-1, -1]
        self.mat_ref2pt = nn.parameter.Parameter(torch.from_numpy(mat_ref2pt))
        self.mat_ref2pt.requires_grad = False

    def ref2pt(self, vec):
        '''
        Map points from reference space to patient space
        '''
        vec2d = vec.reshape(-1, 2)
        pt_vec2d = vec2d @ self.mat_ref2pt[:2, :2] + self.mat_ref2pt[:2, 2]
        return pt_vec2d.ravel()

    def warp(self):
        '''
        warp model in the patient space
        '''
        return self.ref2pt(self.pca.warp(self.coefs))

    def parameters(self):
        return [self.coefs]

    def forward(self, heatmap):
        warped = self.pca.unravel(self.warp())
        probas = []
        for ch, pts in zip(heatmap, warped):
            probas.append(lerp2d(ch, pts))
        return probas


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


image_type = 'lateral_whole'
# image_type = 'ap_whole'
data_dir = Path('../../data/annotation/images') / image_type
outdir = data_dir
skipped = []
l2_reg = .01
pca_models = [(name, TorchPCA(f'pca_{image_type}_{name}.npz'))
              for name in ('L4', 'L5')]
for fn in tqdm(sorted((data_dir).glob('*.json'))):
    file_id = fn.stem

    heatmaps, h = mhd.read(str(data_dir / f'{file_id}.mha'))
    heatmaps = heatmaps.transpose(2, 0, 1)
    anchor_maps = [heatmaps[i] for i in ANCHOR_INDICES]
    heatmaps = heatmaps[:4]

    top_anchor, bottom_anchor = calculate_anchors(anchor_maps)

    results = []
    histories = []
    for name, pca in pca_models:
        n_comps = pca.calc_n_comps(CONTR_RATE)
        pt_pca = PointAlignModel(pca, n_comps, top_anchor, bottom_anchor)
        pt_pca.to(device)
        optimizer = Adam([pt_pca.coefs], lr=LR)

        try:
            for blur_sigma in [10, 1]:
                if blur_sigma == 0:
                    blurred_hm = heatmaps
                else:
                    blurred_hm = np.array(
                        [gaussian_filter(hm, blur_sigma) for hm in heatmaps])
                blurred_hm = (blurred_hm / blurred_hm.max()).astype(np.float32)
                blurred_hm = np.clip(blurred_hm, EPSILON, None)
                t_heatmaps = torch.from_numpy(blurred_hm).to(device)
                t_heatmaps.requires_grad = True

                history = []
                best_loss = 1e100
                best_coefs = 0
                patience = 0
                for i in range(MAX_ITER):
                    probas = pt_pca(t_heatmaps)
                    loss = -torch.log(torch.concat(probas)).mean(
                    ) + l2_reg * torch.linalg.norm(pt_pca.coefs)
                    history.append(loss.detach().item())
                    if history[-1] <= best_loss:
                        best_loss = history[-1]
                        best_coefs = pt_pca.coefs.detach()
                        patience = 0
                    else:
                        patience += 1
                    if patience >= PATIENCE:
                        break
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if SAVE_HISTORY:
                    history = pd.DataFrame(history, columns=['loss'])
                    history['x'] = np.arange(len(history))
                    history['name'] = name
                    history['sigma'] = blur_sigma
                    history['model'] = f'{name}-{blur_sigma}'
                    histories.append(history)
        except IndexError:  # updated coefs become really big when warped points completely miss the heatmaps
            continue
        results.append((name, best_loss, best_coefs))
    if SAVE_HISTORY:
        import seaborn as sns
        df_history = pd.concat(histories, ignore_index=True)
        sns.set(context='notebook', style='whitegrid', font='serif')
        sns.lineplot(data=df_history, x='x', y='loss', hue='model')
        plt.show()

    if len(results) == 0:
        skipped.append(file_id)
        continue
    with open(str(data_dir / f'{file_id}.json')) as f:
        original_json = json.load(f)
    result_json = original_json.copy()
    results.sort(key=lambda e: e[1])  # sort results by loss
    best_name, best_loss, best_coefs = results[0]
    pca = TorchPCA(f'pca_{image_type}_{best_name}.npz')
    pt_pca = PointAlignModel(pca, pca.calc_n_comps(CONTR_RATE), top_anchor,
                             bottom_anchor).to(device)
    warped = pca.unravel(pt_pca.ref2pt(pca.warp(best_coefs.to(device))))
    s = original_json['imageHeight'] / IMG_HEIGHT
    scaled_warped = [(s * w.detach().cpu().numpy()).tolist() for w in warped]

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
if len(skipped) > 0:
    with open(f'skipped_{image_type}.txt', 'w') as f:
        f.write('\n'.join(skipped))

# %%
