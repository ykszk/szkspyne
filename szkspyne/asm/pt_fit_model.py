#%%
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Union

import logzero
import numpy as np
import torch
from joblib import Parallel, delayed
from logzero import logger
from PIL import Image
from scipy.ndimage import gaussian_filter
from szkmipy import boundingbox, mhd
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from szkspyne import labelme_utils, point_utils, settings
settings.set_neck_settings()
from szkspyne.labelme_utils import LabelMe, ShapeDict
from szkspyne.pca_spine import PointsPCA
from szkspyne.torch_utils import bicubic, lerp2d
from szkspyne.asm.point_data import MovablePoints



log_format = '%(color)s[%(levelname)1.1s %(asctime)s.%(msecs)d %(module)s:%(lineno)d]%(end_color)s %(message)s'
formatter = logzero.LogFormatter(fmt=log_format)
logzero.setup_default_logger(formatter=formatter)

#%%

from szkspyne.settings import LABEL_NAMES, ANCHOR_LABEL_NAMES
ANCHOR_LABEL_NAMES_INDICES = [LABEL_NAMES.index(l) for l in ANCHOR_LABEL_NAMES]
EPSILON = np.finfo(float).eps


def trim_bottom_right(arr: np.ndarray, margin):
    '''
    Crop top-left corner of 2d images.
    When input is 3d ndarray with (ch, y, x) dims, bottom-right empty spaces in (y, x) dims are removed.
    Likewise, when input is 4d ndarray with (batch, ch, y, x) dims, bottom-right spaces are removed while (batch, ch) dims are kept intact.

    Args:
        arr: ndarray
    '''
    n_keep_dims = arr.ndim - 2
    xy_bbox = boundingbox.bbox(arr.max(axis=tuple(range(n_keep_dims))) > 0)
    bbox_max = np.concatenate([arr.shape[:n_keep_dims], xy_bbox[1]], axis=0)
    topleft_bbox = (np.zeros_like(bbox_max), bbox_max)
    return boundingbox.crop(arr, topleft_bbox, margin=margin)


class TorchPCA(nn.Module):
    def __init__(self, filename):
        super().__init__()
        with np.load(filename) as a:
            pca = {k: a[k] for k in a.keys()}

        self.pca = pca
        self.labels = [str(l) for l in pca['labels']]
        self.model_name = str(pca['model_name'])
        if 'point_format' in pca and pca['point_format'] != 'xy':
            raise NotImplementedError
        self.mean = nn.parameter.Parameter(
            torch.from_numpy(np.flip(pca['mean'].reshape(-1, 2), axis=1).ravel().astype(np.float32)),
            requires_grad=False,
        )
        components = pca['components']
        components = np.flip(components.reshape(components.shape[0], -1, 2), axis=-1).reshape(components.shape)
        self.components = torch.from_numpy(components.astype(np.float32))
        scaled_components = components * np.sqrt(pca['explained_variance'])[:, np.newaxis]
        self.scaled_components = nn.parameter.Parameter(
            torch.from_numpy(scaled_components.astype(np.float32)), requires_grad=False
        )
        point_counts = pca['point_counts']
        self.split = [0] + np.cumsum(2 * point_counts).tolist()

        self.template: MovablePoints = MovablePoints(
            points=np.flip(pca['template_points'].reshape(-1, 2), axis=1),
            reference_points=pca['template_reference'].tolist(),
            point_counts=point_counts.tolist(),
        )

    def calc_n_comps(self, contr_rate) -> int:
        '''
        calculate the number of components required for the specified contribution ratio
        '''
        cs = np.cumsum(self.pca['explained_variance_ratio'])
        idx = np.nonzero(cs >= contr_rate)[0][0]
        return idx + 1

    def warp(self, coefs):
        '''
        warp model in the reference space
        '''
        # logger.info('%s, %s, %s', self.mean.device, coefs.device, self.scaled_components.device)
        return self.mean + torch.sum(torch.unsqueeze(coefs, 1) * self.scaled_components[: len(coefs)], dim=0)

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


#%%
npz_file = '/poseidon/users/suzuki/projects/scoliosis/result/neck_manual_20250620/folds/0/neck_full.npz'
torch_pca = TorchPCA(npz_file)
torch_pca.template

#%%
from szkspyne.labelme_utils import LabelMe, ShapeDict
from szkspyne.asm.point_data import MovablePoints, SpinePointData
lm_file = '/poseidon/users/suzuki/projects/scoliosis/result/neck_manual_20250620/folds/0/first.json'
lm_data = LabelMe.from_file(lm_file)

#%%
from szkmipy import mhd
import matplotlib.pyplot as plt

vol, h = mhd.read('/poseidon/users/suzuki/projects/scoliosis/result/neck_manual_20250620/folds/0/first.mha')
mip = np.max(vol, axis=0)

#%%
image_scale = mip.shape[0] / lm_data.imageHeight
scaled_lm_data = lm_data.scale(image_scale)
spine_point_data = SpinePointData.from_shape_dict(scaled_lm_data.into_shape_dict())

#%%
# spine_point_data.reference_points
# tf_ref2ind = torch_pca.template.calculate_transform_with_missing(spine_point_data, inverse=True)
# tf_mat = tf_ref2ind.params

# plt.imshow(mip, cmap='gray')

# ones
# coefs = torch.ones(2 * len(torch_pca.labels), dtype=torch.float32, device=torch_pca.mean.device)
# mean_points = torch_pca.unravel(torch_pca.warp(coefs))
# for i, label in enumerate(torch_pca.labels):
#     points = mean_points[i]
#     points = points.detach().cpu().numpy()
#     points = np.flip(points, axis=1)  # (x, y)
#     plt.scatter(points[:, 0], points[:, 1], s=10, label=label)
# plt.legend()
# plt.axis('off')
# plt.show()



#%%
# plt.imshow(mip, cmap='gray')
# # apply transform
# tfed_mean_points = []
# pt_tf_mat = torch.from_numpy(tf_mat).to(torch_pca.mean.device).float()
# for i, label in enumerate(torch_pca.labels):
#     points = mean_points[i]
#     # points = torch_pca.template.ref2pt(points)
#     # points = points.detach().cpu().numpy()
#     # points = np.flip(points, axis=1)  # (x, y)

#     # xs = tf_mat[0, :2] @ points.T + tf_mat[0, 2]
#     # ys = tf_mat[1, :2] @ points.T + tf_mat[1, 2]
#     ys = pt_tf_mat[1, :2].flip(dims=(0,)) @ points.T + pt_tf_mat[1, 2]
#     xs = pt_tf_mat[0, :2].flip(dims=(0,)) @ points.T + pt_tf_mat[0, 2]
#     points = np.stack([xs, ys], axis=1)

#     tfed_mean_points.append(points)
#     plt.scatter(points[:, 0], points[:, 1], s=10, label=label)
# plt.legend()
# plt.axis('off')
# plt.show()

#%%
class PointAlignModel(nn.Module):
    def __init__(self, pca: TorchPCA, n_comps: int, points: MovablePoints, use_bicubic=False):
        super().__init__()
        self.coefs = nn.parameter.Parameter(torch.zeros([n_comps], dtype=torch.float32, requires_grad=True))
        self.pca = pca
        self.use_bicubic = use_bicubic
        self.points = points


        # convert transformation matrix in format (x, y) to (y, x)
        def trans_mat(mat_xy):
            yx_mat = np.zeros_like(mat_xy)
            yx_mat[2, 2] = 1
            yx_mat[:2, :2] = np.flip(mat_xy[:2, :2], axis=(0, 1))
            yx_mat[0, 2], yx_mat[1, 2] = mat_xy[1, 2], mat_xy[0, 2]
            return yx_mat


        tf = pca.template.calculate_transform_with_missing(points, inverse=True)
        self.mat_ref2ind = nn.parameter.Parameter(torch.from_numpy(trans_mat(tf.params.astype(np.float32))), requires_grad=False)
        self.mat_ind2ref = nn.parameter.Parameter(torch.from_numpy(trans_mat(np.linalg.inv(tf.params).astype(np.float32))), requires_grad=False)

    def resize_n_comps(self, n_comps: int):
        '''
        Change the number of components used in the model
        '''
        if n_comps == len(self.coefs):
            return
        if n_comps > len(self.coefs):
            new_coefs = torch.zeros([n_comps], dtype=torch.float32, requires_grad=False)
            new_coefs[:len(self.coefs)] = self.coefs.detach().clone()
            self.coefs = nn.parameter.Parameter(new_coefs, requires_grad=True)
        else:
            self.coefs = nn.parameter.Parameter(self.coefs[:n_comps], requires_grad=True)

    def _apply_transform(self, vec, mat):
        '''
        Apply transformation matrix to the vector

        Args:
            vec: 2d tensor of (y, x) points
            mat: 2d tensor of transformation matrix
        '''
        vec2d = vec.reshape(-1, 2)
        transformed_vec = mat[:2, :2] @ vec2d.T + mat[:2, 2].unsqueeze(1)
        return transformed_vec.T.ravel()

    def ref2ind(self, vec):
        '''
        Map points from reference space to individual space

        Args:
            vec: 2d tensor of (y, x) points
        '''
        return self._apply_transform(vec, self.mat_ref2ind)

    def ind2ref(self, vec):
        '''
        Map points from individual space to reference space
        '''
        return self._apply_transform(vec, self.mat_ind2ref)

    def warp(self):
        '''
        warp model in the individual space
        '''
        return self.ref2ind(self.pca.warp(self.coefs))

    def reduce(self, vec: torch.Tensor):
        '''
        Calculate and update pc coefficients

        Args:
            vec: 2d tensor of (y, x) points in reference space
        '''
        ref_points = self.ind2ref(vec.ravel())
        ref_points = ref_points  # .detach().cpu().numpy()
        # ref_points = torch.flip(ref_points, dims=(1,)) # (y, x) to (x, y)
        flat_points = ref_points.ravel() - self.pca.mean
        projs = torch.matmul(flat_points, self.pca.components.T)  # pca['components'])
        projs = projs / torch.sqrt(torch.Tensor(self.pca.pca['explained_variance']))
        self.coefs = torch.nn.parameter.Parameter(torch.Tensor(projs[: len(self.coefs)]))

    def parameters(self):
        return [self.coefs]

    def forward(self, heatmap):
        warped = self.pca.unravel(self.warp())
        probas = []
        assert len(heatmap) == len(warped)
        for ch, pts in zip(heatmap, warped):
            if self.use_bicubic:
                probas.append(lerp2d(ch, pts))
            else:
                probas.append(bicubic(ch, pts))
        return probas

    def icp_loss(self, points):
        warped = self.pca.unravel(self.warp())
        distances = []
        assert len(points) == len(warped)
        for ref, pts in zip(points, warped):
            m_dist = torch.cdist(ref, pts, p=2)
            if len(pts) >= len(ref):
                # `dim=1` to only count ref to pts distances
                min_dists = torch.min(m_dist, dim=1, keepdim=False)[0]  # [0] to ignore returned indices
            else:
                min_dists = torch.min(m_dist, dim=0, keepdim=False)[0]
            distances.append(min_dists)
        return distances

    def icp(self, t_points: List[torch.Tensor], lr, max_iter, patience):
        '''
        t_points: list of (y, x) coordinates in patient space
        '''
        optimizer = Adam([self.coefs], lr=lr, weight_decay=0.1)

        history = []
        best_loss = np.finfo(np.float64).max
        acc_patience = 0
        for i in range(max_iter):
            distances = self.icp_loss(t_points)
            loss = torch.concat(distances, dim=0).mean()
            history.append(loss.detach().item())
            if history[-1] <= best_loss:
                best_loss = history[-1]
                acc_patience = 0
            else:
                acc_patience += 1
            if acc_patience >= patience:
                logger.debug('Early stopping ICP at %d %f', i, history[-1])
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.detach()

# first_n_comps = torch_pca.calc_n_comps(0.80)
# print('first_n_comps:', first_n_comps)
# align_model = PointAlignModel(torch_pca, first_n_comps, spine_point_data, use_bicubic=False)
# align_model.to(torch.device('cpu'))

# shape_dict = scaled_lm_data.into_shape_dict()
# vert_points = [np.flip(shape_dict.shapes['point'][l], -1)[:, 0].copy() for l in torch_pca.labels]
# t_vert_points = [torch.from_numpy(p).float() for p in vert_points]


# icp_loss_value = align_model.icp(t_vert_points, lr=0.1, max_iter=100, patience=10)
# second_n_comps = torch_pca.calc_n_comps(0.95)
# print('n_comps:', second_n_comps)
# align_model.resize_n_comps(second_n_comps)
# #%%
# from PIL import Image

# print(align_model.coefs)
# icp_points = align_model.pca.unravel(align_model.warp())
# bg_path = '/poseidon/users/suzuki/projects/scoliosis/result/neck_manual_20250620/folds/0/bg.jpg'
# bg = Image.open(bg_path)
# bg = bg.convert('L')
# bg.thumbnail((100000, mip.shape[0]), Image.Resampling.BILINEAR)

# plt.imshow(bg, cmap='gray')
# plt.imshow(mip, alpha=0.5)
# plt.title('Heatmap overlay')

# #%%

# plt.imshow(bg, cmap='gray')

# for i, label in enumerate(torch_pca.labels):
#     points = icp_points[i]
#     points = points.detach().cpu().numpy()
#     # points = vert_points[i]
#     points = np.flip(points, axis=1)  # (x, y)
#     plt.scatter(points[:, 0], points[:, 1], s=10, label=label)
# # plt.legend()
# plt.axis('off')
# plt.title('After ICP')
# plt.show()

# #%%
# plt.imshow(bg, cmap='gray')
# for label in torch_pca.labels:
#     points = shape_dict.shapes['point'][label]
#     points = np.array(points)[:, 0]
#     plt.scatter(points[:, 0], points[:, 1], s=10, label=label)
# plt.legend()
# plt.axis('off')
# plt.title('Heatmap points')
# plt.show()


#%%


def fit(args, pca_models: List[TorchPCA], original_json: LabelMe, heatmaps: np.ndarray, heatmap_height: int):
    logger.setLevel(os.environ.get('LOGLEVEL', 'INFO').upper())
    # kps = measure_utils.get_point_dict(original_json)
    s = heatmap_height / original_json.imageHeight  # scale to heatmap
    scaled_lm_data = original_json.scale(s)
    shape_dict: ShapeDict = scaled_lm_data.into_shape_dict()
    # asm_points = [np.flip(shape_dict.shapes['point'][l], 2)[0].copy() for l in settings.ASM_LABEL_NAMES]  # (x, y) to (y, x)
    asm_points = []
    for label in pca_models[0].labels:
        if label in shape_dict.shapes['point']:
            asm_points.append(np.flip(shape_dict.shapes['point'][label], 2)[:, 0].copy())
        else:
            logger.warning('No point for %s. Using empty point.', label)
            asm_points.append(np.empty((0, 2), dtype=np.float32))  # empty point for missing labels

    logger.debug('asm_points: %s', asm_points)
    asm_shapes = [a.shape for a in asm_points]
    logger.debug('asm_shapes: %s', asm_shapes)
    asm_points = [vp[vp[:, 0].argsort()] for vp in asm_points]  # sort by y
    asm_points_count = np.array([len(p) for p in asm_points])
    logger.debug('asm_points_count: %s', asm_points_count)
    if args.skip_complete:
        for pca in pca_models:
            if np.all(asm_points_count == np.array(pca.pca['point_counts'])):
                logger.debug('Skip completed set for %s', pca.model_name)
                # kps = measure_utils.get_point_dict(original_json)
                shape_dict: ShapeDict = original_json.into_shape_dict()
                kps = shape_dict.shapes['point']
                for label in kps.keys():
                    kps[label] = np.flip(kps[label], axis=1)  # xy -> yx because kps2shapes's input is in yx format
                [kps.pop(label, None) for label in ANCHOR_LABEL_NAMES]
                original_json.shapes = labelme_utils.kps2shapes(list(kps.values()), list(kps.keys()))
                original_json.flags[pca.model_name] = True
                original_json.flags['PCA'] = False
                # original_json['pca'] = {}

                return original_json
    # heatmaps = trim_bottom_right(heatmaps, 4)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    results = []
    histories = []
    d_t_heatmaps = {}
    for blur_sigma in args.sigma:
        if blur_sigma == 0:
            blurred_hm = heatmaps[: len(pca_models[0].labels)]
        else:
            blurred_hm = np.array(
                [gaussian_filter(hm, blur_sigma) for hm in heatmaps[: len(pca_models[0].labels)]]
            )
        blurred_hm = (blurred_hm / settings.PROB_MAX).astype(np.float32)
        blurred_hm = np.clip(blurred_hm, 0.001, None)
        t_heatmaps = torch.FloatTensor(blurred_hm).to(device)
        t_heatmaps.requires_grad = True
        d_t_heatmaps[blur_sigma] = t_heatmaps

    for pca in pca_models:
        # Skip model if detected points exceed the number of model's points
        if args.skip_model and np.any(np.array([len(p) for p in asm_points]) > np.array(pca.pca['point_counts'])):
            logger.debug('Skip model: %s', pca.model_name)
            continue
        name = pca.model_name
        icp_n_comps = pca.calc_n_comps(args.icp_contr_rate)
        movable_points = SpinePointData.from_shape_dict(scaled_lm_data.into_shape_dict())
        pt_pca = PointAlignModel(pca, icp_n_comps, movable_points, use_bicubic=args.bicubic)
        pt_pca.to(device)
        # t_vert_points = [pt_pca.pt2ref(torch.Tensor(p).to(device)) for p in vert_points]
        t_vert_points = [torch.Tensor(p).to(device) for p in asm_points]
        if np.all(asm_points_count == np.array(pca.pca['point_counts'])):
            pt_pca.reduce(torch.concatenate(t_vert_points, dim=0))
            histories.append(
                {'name': f'{name}-reduce', 'history': [], 'coefs': pt_pca.coefs.detach().numpy().tolist()}
            )
            results.append((name, 100, pt_pca.coefs.detach().cpu(), pca, []))
            # continue
        else:
            best_loss = pt_pca.icp(t_vert_points, args.icp_lr, args.max_iter, args.patience)
            histories.append(
                {
                    'name': f'{name}-icp',
                    'history': [best_loss.detach().cpu().item()],
                    'coefs': pt_pca.coefs.detach().cpu().numpy().tolist(),
                }
            )
            results.append((name, best_loss, pt_pca.coefs.detach().cpu(), pca, []))
        n_comps = pca.calc_n_comps(args.contr_rate)
        pt_pca.resize_n_comps(n_comps)
        pt_pca.to(device) # TODO:
        optimizer = Adam([pt_pca.coefs], lr=args.lr, weight_decay=args.l2_reg)
        try:
            for blur_sigma in args.sigma:
                logger.debug('Processing %s with blur sigma %s', name, blur_sigma)
                t_heatmaps = d_t_heatmaps[blur_sigma].to(device)
                history = []
                best_loss = np.finfo(np.float64).max
                best_coefs = None
                best_pca = None
                patience = 0
                for i in range(args.max_iter):
                    probas = pt_pca(t_heatmaps)
                    loss_hm = -torch.log(torch.concat(probas)).mean()
                    distances = pt_pca.icp_loss(t_vert_points)
                    loss_icp = torch.concat(distances, dim=0).mean()
                    loss = loss_hm + args.icp_weight * loss_icp
                    history.append(loss.detach().cpu().item())
                    if len(history) == 1 or history[-1] <= best_loss:
                        best_loss = history[-1]
                        best_coefs = pt_pca.coefs.detach().cpu()
                        best_probas = [p.detach().cpu() for p in probas]
                        best_pca = pca
                        patience = 0
                    else:
                        patience += 1
                    if patience >= args.patience:
                        break
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                assert best_coefs is not None
                if True:  # save history
                    d_history = {}
                    d_history['name'] = name
                    d_history['sigma'] = blur_sigma
                    d_history['history'] = [None if np.isnan(h) else h for h in history]
                    d_history['coefs'] = best_coefs.numpy().tolist()
                    histories.append(d_history)
        except IndexError:  # updated coefs become really big when warped points completely miss the heatmaps
            logger.warning('Index error')
            continue
        results.append((name, best_loss, best_coefs, best_pca, best_probas))

    if len(results) == 0:
        raise RuntimeError("No result found")
    result_json = original_json.model_copy()
    results.sort(key=lambda e: e[1])  # sort results by loss
    best_name, best_loss, best_coefs, best_pca, best_probas = results[0]
    logger.debug('results: %s', results)
    logger.debug('best loss: %s', best_loss)
    pca = best_pca
    pt_pca = PointAlignModel(pca, pca.calc_n_comps(args.contr_rate), movable_points, use_bicubic=args.bicubic).to(device)
    warped = pca.unravel(pt_pca.ref2ind(pca.warp(best_coefs.to(device))))
    s = original_json.imageHeight / heatmap_height
    scaled_warped = [(s * w.detach().cpu().numpy()).tolist() for w in warped]

    # original_shapes = result_json.shapes
    # for l in pca.labels + settings.ANCHOR_LABEL_NAMES:
    #     original_shapes = list(
    #         filter(lambda shape: shape['label'] != l or shape['shape_type'] != 'point', original_shapes)
    #     )
    result_shapes = labelme_utils.kps2shapes(scaled_warped, pca.labels)
    result_shapes = [labelme_utils.LabelMe.Shape.from_dict(s) for s in result_shapes]
    # result_shapes.extend(original_shapes)
    result_json.shapes = result_shapes
    result_json.flags = {best_name: True, 'PCA': True}
    if args.history:
        result_json['pca'] = histories
    return result_json

#%%

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fit PCA point model to the given heatmap.')
    parser.add_argument(
        'input',
        help='Either (1) an input data directory containing json and mha pairs or (2) an input njson and a mha',
        metavar='<input>',
        nargs='+',
    )
    parser.add_argument('--output', help='Output directory. Use input directory by default.')
    parser.add_argument('-m', '--model', help='PCA model filename(s).', nargs='+')
    parser.add_argument('--sigma', help='Sigma value(s) for heatmap blurring.', nargs='+', type=float, default=[10, 1])
    parser.add_argument('--lr', help='Initial learning rate. default: %(default)s', type=float, default=0.1)
    parser.add_argument(
        '--icp_lr', help='Initial learning rate for ICP. default: %(default)s', type=float, default=0.1
    )
    parser.add_argument('--icp_weight', help='Weight for ICP loss. default: %(default)s', type=float, default=0.01)
    parser.add_argument(
        '--max_iter', help='The maximum number of iteration for fitting. default: %(default)s', type=int, default=200
    )
    parser.add_argument('--patience', help='Patience for early stopping. default: %(default)s', type=int, default=20)
    parser.add_argument(
        '--contr_rate', help='Contribution rate for PCA. default: %(default)s', type=float, default=0.99
    )
    parser.add_argument(
        '--icp_contr_rate', help='Contribution rate for initial ICP. default: %(default)s', type=float, default=0.80
    )
    parser.add_argument(
        '--l2_reg',
        help='Coefficient for the L2 regularization term (weight decay). default: %(default)s',
        type=float,
        default=0.01,
    )
    parser.add_argument('--skipped', help='Optional argument', metavar='<name>')
    parser.add_argument('--device', help='Device for pytorch')
    parser.add_argument('--channel_first', help='Heatmap is in channel-first format', action='store_true')
    parser.add_argument('--history', help='Save optimization history and pca parameters', action='store_true')
    parser.add_argument(
        '--bicubic', help='Use bicubic interpolation instead of bilinear interpolation', action='store_true'
    )
    parser.add_argument(
        '--skip_complete',
        help='Skip fitting if given set of points is complete (i.e. the number of points matches expected number of points).',
        action='store_true',
    )
    parser.add_argument(
        '--skip_model',
        help='Skip model that has less points than the number of points in the json.',
        action='store_true',
    )
    parser.add_argument(
        '--threads', help='Integer value for torch.set_num_threads(). default: %(default)d', type=int, default=1
    )
    parser.add_argument('--jobs', help='The number of jobs. default: %(default)d', type=int, default=None)

    logger.setLevel(os.environ.get('LOGLEVEL', 'INFO').upper())
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning('Unknown arguments:%s', unknown)

    # The operator 'aten::index.Tensor' is not current implemented for the MPS device.
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    torch.set_num_threads(args.threads)

    channel_first = args.channel_first
    pca_models = [TorchPCA(fn) for fn in args.model]


    n_jobs = args.jobs
    skipped = []

    if len(args.input) == 1:
        logger.info('Directory input')
        data_dir = Path(args.input[0])
        outdir = data_dir
        if args.output:
            outdir = Path(args.output)
            logger.info('Output directory: %s', outdir)

        def do_dirin(fn: Path):
            logger.setLevel(os.environ.get('LOGLEVEL', 'INFO').upper())
            torch.set_num_threads(args.threads)
            file_id = fn.stem
            logger.debug('Processing %s', file_id)
            heatmaps, h = mhd.read(str(data_dir / f'{file_id}.mha'))
            if not channel_first:
                heatmaps = heatmaps.transpose(2, 0, 1)
            heatmap_height = heatmaps.shape[1]
            with open(str(data_dir / f'{file_id}.json')) as f:
                original_json = json.load(f)
            original_json = LabelMe.model_validate(original_json)
            try:
                result = fit(args, pca_models, original_json, heatmaps, heatmap_height)
            except Exception as e:
                # import traceback
                # return f"{file_id},{traceback.format_exc()}"
                logger.error('Error processing %s: %s', file_id, e)
                return f"{file_id},{e}"
            with open(outdir / f'{file_id}.json', 'w') as f:
                f.write(result.model_dump_json())
                # json.dump(result_json, f, indent=2)
            
            return None

        fns = sorted((data_dir).glob('*.json'))
        logger.info('Fit')
        # skipped = Parallel(n_jobs=n_jobs)(delayed(do_dirin)(fn) for fn in tqdm(fns))
        skipped = [do_dirin(fn) for fn in tqdm(fns, total=len(fns))]
        if skipped:  # remove None
            skipped = [s for s in skipped if s]
        logger.info('Done')

    elif len(args.input) == 2:
        logger.info('Ndjson input')
        ndjson_fn = Path(args.input[0])
        hm_filename = Path(args.input[1])
        if args.output is None or not args.output.endswith('.ndjson'):
            logger.error('Output ndjson should be specified for ndjson input')
        if ndjson_fn.suffix != '.ndjson':
            logger.error('Invalid input ndjson filename', ndjson_fn.name)
        logger.info('load image heatmaps from %s', hm_filename)
        # header = mhd.read_header(hm_filename)
        heatmap_iter, header = mhd.read_iterator(hm_filename)
        heatmap_dims = mhd.get_dim(header)
        if channel_first:
            heatmap_height = heatmap_dims[1]
        else:
            heatmap_height = heatmap_dims[2]
        logger.info('Heatmap height: %d', heatmap_height)
        all_heatmaps = []
        for heatmap in tqdm(heatmap_iter, total=heatmap_dims[-1]):
            if not channel_first:
                heatmap = heatmap.transpose(2, 0, 1)
            all_heatmaps.append(trim_bottom_right(heatmap, margin=32))
        logger.info('Heatmaps loaded: %d', len(all_heatmaps))
        # all_heatmaps, h = mhd.read(hm_filename)
        # logger.info('Heatmaps shape: %s', all_heatmaps.shape)
        # heatmap_height = all_heatmaps.shape[1]
        # if not channel_first:
        #     logger.info('Transpose heatmaps to channel-first format')
        #     all_heatmaps = all_heatmaps.transpose(0, 3, 1, 2)
        # # all_heatmaps = all_heatmaps[:, :len(pca_models[0].labels)]
        # logger.info('trim bottom right of heatmaps')
        # # all_heatmaps = trim_bottom_right(all_heatmaps, margin=32).copy()
        # all_heatmaps = [trim_bottom_right(hm.copy(), margin=32) for hm in all_heatmaps]

        ndjson = []
        with open(ndjson_fn) as f:
            for line in f:
                ndjson.append(json.loads(line))

        def do(original_json, heatmaps):
            torch.set_num_threads(args.threads)
            file_id = original_json['filename']
            try:
                result = fit(args, pca_models, LabelMe.model_validate(original_json['content']), heatmaps, heatmap_height)
            except Exception as e:
                return f"{file_id},{e}"
            result_json = result.model_dump()
            return {'filename': file_id, 'content': result_json}

        logger.info('Fit')
        result_skipped = Parallel(n_jobs=n_jobs)(
            delayed(do)(*args) for args in tqdm(zip(ndjson, all_heatmaps), total=len(ndjson))
        )
        assert result_skipped is not None
        results = [r for r in result_skipped if isinstance(r, dict)]
        logger.info('Save ndjson: %s', len(results))
        with open(args.output, 'w') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')))
                f.write('\n')
        logger.info('Done')
        skipped = [r for r in result_skipped if isinstance(r, str)]
    else:
        print('Invalid input')
        return 1

    assert skipped is not None
    if args.skipped and len(skipped) > 0:
        print(f'Save skipped {len(skipped)} files', args.skipped)
        with open(args.skipped, 'w') as f:
            f.write('\n'.join(skipped))
    else:
        print('No skipped files')
    return 0


if __name__ == '__main__':
    sys.exit(main())

#%%
sigma = np.sqrt(5)
peak_value = 1 / np.sqrt(2 * np.pi * sigma**2)
peak_value