import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Union

import logzero
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from joblib import Parallel, delayed
from logzero import logger
from PIL import Image
from scipy.ndimage import gaussian_filter
from szkmipy import boundingbox, mhd
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

import szkspyne.utils.measure as measure_utils
from szkspyne import labelme_utils, point_utils, settings
from szkspyne.pca_spine import PointsPCA
from szkspyne.torch_utils import bicubic, lerp2d


log_format = '%(color)s[%(levelname)1.1s %(asctime)s.%(msecs)d %(module)s:%(lineno)d]%(end_color)s %(message)s'
formatter = logzero.LogFormatter(fmt=log_format)
logzero.setup_default_logger(formatter=formatter)


ANCHOR_LABELS = ['C7-TL', 'S-TL']
ANCHOR_INDICES = [settings.LABEL_NAMES.index(l) for l in ANCHOR_LABELS]
EPSILON = np.finfo(float).eps


def trim_bottom_right(arr: np.ndarray, margin):
    '''
    Args:
        arr: 3d ndarray with (ch, y, x) dims. ch dimension will be kept intact
    '''
    bbox = boundingbox.bbox(arr.max(axis=0)>0)
    bbox = (np.insert(bbox[0], [0], 0), np.insert(bbox[1], [0], arr.shape[0]-1)) # no cropping in channel axis
    topleft_bbox = (np.zeros_like(bbox[0]), bbox[1])
    return boundingbox.crop(arr, topleft_bbox, margin=margin)


class TorchPCA(nn.Module):

    def __init__(self, filename):
        super().__init__()
        with np.load(filename) as a:
            pca = {k: a[k] for k in a.keys()}

        self.pca = pca
        self.labels = [str(l) for l in pca['labels']]
        self.model_name = str(pca['model_name'])
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

    def __init__(self, pca: TorchPCA, n_comps: int, top_anchor, bottom_anchor, use_bicubic=False):
        super().__init__()
        self.coefs = nn.parameter.Parameter(
            torch.zeros([n_comps], dtype=torch.float32, requires_grad=True))
        self.pca = pca
        self.top_anchor = top_anchor
        self.bottom_anchor = bottom_anchor
        self.use_bicubic = use_bicubic
        mat_ref2pt = PointsPCA(pca.pca['ref_height']).to_target_mat(
            top_anchor[::-1], bottom_anchor[::-1]).astype(np.float32)
        # convert (x, y) to (y, x)
        mat_ref2pt[0, 0], mat_ref2pt[1, 1] = mat_ref2pt[1, 1], mat_ref2pt[0, 0]
        t = mat_ref2pt[:2, 2].copy()
        mat_ref2pt[0, 2], mat_ref2pt[1, 2] = t[1], t[0]
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
            # `dim=1` to only count ref to pts distances
            min_dists = torch.min(m_dist, dim=1, keepdim=False)[0] # [0] to ignore returned indices
            distances.append(min_dists)
        return distances

    def icp(self, t_points: List[torch.Tensor], lr, max_iter, patience):
        '''
        t_points: list of (y, x) coordinates
        '''
        optimizer = Adam([self.coefs],
                            lr=lr, weight_decay=0.1)

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
    
def calculate_anchors(anchor_maps):
    '''
    Args:
        anchor_maps: [ch, y, x]
    '''

    anchor_candidates = point_utils.local_center_of_masses(anchor_maps)
    anchors = []
    for candidates, hm in zip(anchor_candidates, anchor_maps):
        if len(candidates) == 0:
            anchors.append(None)
        if len(candidates) == 1:
            anchors.append(candidates[0])
        else:
            probas = [hm[int(p[0]), int(p[1])] for p in candidates]
            anchors.append(candidates[np.argmax(probas)])

    return anchors


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Fit PCA point model to the given heatmap.')
    parser.add_argument(
        'input',
        help=
        'Either (1) an input data directory containing json and mha pairs or (2) an input njson and a mha',
        metavar='<input>',
        nargs='+')
    parser.add_argument(
        '--output', help='Output directory. Use input directory by default.')
    parser.add_argument('-m',
                        '--model',
                        help='PCA model filename(s).',
                        nargs='+')
    parser.add_argument('--sigma',
                        help='Sigma value(s) for heatmap blurring.',
                        nargs='+',
                        type=float,
                        default=[10, 1])
    parser.add_argument('--lr',
                        help='Initial learning rate. default: %(default)s',
                        type=float,
                        default=0.1)
    parser.add_argument('--icp_lr',
                        help='Initial learning rate for ICP. default: %(default)s',
                        type=float,
                        default=0.1)
    parser.add_argument(
        '--max_iter',
        help=
        'The maximum number of iteration for fitting. default: %(default)s',
        type=int,
        default=200)
    parser.add_argument(
        '--patience',
        help='Patience for early stopping. default: %(default)s',
        type=int,
        default=20)
    parser.add_argument('--contr_rate',
                        help='Contribution rate for PCA. default: %(default)s',
                        type=float,
                        default=0.999)
    parser.add_argument(
        '--l2_reg',
        help=
        'Coefficient for the L2 regularization term (weight decay). default: %(default)s',
        type=float,
        default=0.01)
    parser.add_argument('--skipped',
                        help='Optional argument',
                        metavar='<name>')
    parser.add_argument('--device', help='Device for pytorch')
    parser.add_argument('--channel_first',
                        help='Heatmap is in channel-first format',
                        action='store_true')
    parser.add_argument('--bicubic',
                        help='Use bicubic interpolation instead of bilinear interpolation',
                        action='store_true')
    parser.add_argument('--skip_complete',
                        help='Skip fitting if given set of points is complete (i.e. the number of points matches expected number of points).',
                        action='store_true')
    parser.add_argument('--skip_model',
                        help='Skip model that has less points than the number of points in the json.',
                        action='store_true')
    parser.add_argument(
        '--threads',
        help='Integer value for torch.set_num_threads(). default: %(default)d',
        type=int,
        default=1)
    parser.add_argument('--jobs',
                        help='The number of jobs. default: %(default)d',
                        type=int,
                        default=4)

    logger.setLevel(os.environ.get('LOGLEVEL', 'INFO').upper())
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning('Unknown arguments:%s', unknown)

    # The operator 'aten::index.Tensor' is not current implemented for the MPS device.
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(args.threads)
    if args.device:
        device = args.device

    logger.info('Device:%s', device)

    channel_first = args.channel_first
    pca_models = [TorchPCA(fn) for fn in args.model]
    h_flip = False

    def fit(original_json: dict, heatmaps: np.ndarray):
        logger.setLevel(os.environ.get('LOGLEVEL', 'INFO').upper())
        kps = measure_utils.get_point_dict(original_json)
        if not channel_first:
            heatmaps = heatmaps.transpose(2, 0, 1)
        heatmap_height = heatmaps.shape[1]
        s = heatmap_height / original_json['imageHeight'] # scale to heatmap
        vert_points = [s * np.flip(kps[l], 1).copy() for l in settings.VERTEBRA_LABEL_NAMES] # (x, y) to (y, x)
        vert_points = [np.sort(vp, axis=0) for vp in vert_points] # sort by y
        vert_points_count = np.array([len(p) for p in vert_points])
        if args.skip_complete:
            for pca in pca_models:
                if np.all(vert_points_count == np.array(pca.pca['point_counts'])):
                    logger.debug('Skip completed set for %s', pca.model_name)
                    kps = measure_utils.get_point_dict(original_json)
                    kps = measure_utils.sort_kps(kps)
                    for label in kps.keys():
                        kps[label] = np.flip(kps[label], axis=1) # xy -> yx because kps2shapes's input is in yx format
                    [kps.pop(label, None) for label in ['C7-TL', 'C7-TR', 'S-TL', 'S-TR']]
                    original_json['shapes'] = labelme_utils.kps2shapes(list(kps.values()), list(kps.keys()))
                    original_json['flags'][pca.model_name] = True
                    original_json['flags']['PCA'] = False
                    original_json['pca'] = {}
                        
                    return original_json
        if h_flip:  #h_flip:  # horizontal flip
            heatmaps = np.flip(heatmaps, axis=2)
        heatmaps = trim_bottom_right(heatmaps, 4)

        anchor_maps = [heatmaps[i] for i in ANCHOR_INDICES]
        top_anchor, bottom_anchor = calculate_anchors(anchor_maps)
        if top_anchor is None or bottom_anchor is None:
            raise RuntimeError("No anchor found")
        t_vert_points = [torch.Tensor(p).to(device) for p in vert_points]

        results = []
        histories = []
        for pca in pca_models:
            # Skip model if detected points exceed the number of model's points
            if args.skip_model and np.any(np.array([len(p) for p in t_vert_points]) > np.array(pca.pca['point_counts'])):
                logger.debug('Skip model: %s', pca.model_name)
                continue
            name = pca.model_name
            n_comps = pca.calc_n_comps(args.contr_rate)
            pt_pca = PointAlignModel(pca, n_comps, top_anchor, bottom_anchor)
            pt_pca.to(device)
            best_loss = pt_pca.icp(t_vert_points, args.icp_lr, args.max_iter, args.patience)
            histories.append({'history':[], 'name':f'{name}-icp', 'coefs': pt_pca.coefs.detach().numpy().tolist()})

            optimizer = Adam([pt_pca.coefs],
                             lr=args.lr,
                             weight_decay=args.l2_reg)
            try:
                for blur_sigma in args.sigma:
                    if blur_sigma == 0:
                        blurred_hm = heatmaps[:len(pca.labels)]
                    else:
                        blurred_hm = np.array([
                            gaussian_filter(hm, blur_sigma)
                            for hm in heatmaps[:len(pca.labels)]
                        ])
                    blurred_hm = (blurred_hm / settings.PROB_MAX).astype(
                        np.float32)
                    blurred_hm = np.clip(blurred_hm, 0.001, None)
                    t_heatmaps = torch.FloatTensor(blurred_hm).to(device)
                    t_heatmaps.requires_grad = True

                    history = []
                    best_loss = np.finfo(np.float64).max
                    best_coefs = None
                    best_pca = None
                    patience = 0
                    for i in range(args.max_iter):
                        probas = pt_pca(t_heatmaps)
                        loss = -torch.log(torch.concat(probas)).mean()
                        history.append(loss.detach().cpu().item())
                        if len(history)==1 or history[-1] <= best_loss:
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
                        d_history['history'] = [None if np.isnan(h) else h for h in history]
                        d_history['name'] = name
                        d_history['sigma'] = blur_sigma
                        d_history['coefs'] = best_coefs.numpy().tolist()
                        histories.append(d_history)
            except IndexError:  # updated coefs become really big when warped points completely miss the heatmaps
                logger.warning('Index error')
                continue
            results.append((name, best_loss, best_coefs, best_pca, best_probas))

        if len(results) == 0:
            raise RuntimeError("No result found")
        result_json = original_json.copy()
        results.sort(key=lambda e: e[1])  # sort results by loss
        best_name, best_loss, best_coefs, best_pca, best_probas = results[0]
        logger.debug('results: %s', results)
        logger.debug('best loss: %s', best_loss)
        pca = best_pca
        pt_pca = PointAlignModel(pca, pca.calc_n_comps(args.contr_rate),
                                 top_anchor, bottom_anchor).to(device)
        warped = pca.unravel(pt_pca.ref2pt(pca.warp(best_coefs.to(device))))
        s = original_json['imageHeight'] / heatmap_height
        scaled_warped = [(s * w.detach().cpu().numpy()).tolist()
                         for w in warped]
        # if h_flip:
        #     scaled_warped = [[(e[0], original_json['imageWidth']-e[1]) for e in sw] for sw in scaled_warped]
        # for sw in scaled_warped:
        #     sw[:, 1] = original_json['imageWidth'] - sw[:, 1]

        original_shapes = result_json['shapes']
        for l in pca.labels + settings.ANCHOR_LABEL_NAMES:
            original_shapes = list(
                filter(
                    lambda shape: shape['label'] != l or shape['shape_type'] !=
                    'point', original_shapes))
        result_shapes = labelme_utils.kps2shapes(scaled_warped, pca.labels)
        result_shapes.extend(original_shapes)
        result_json['shapes'] = result_shapes
        result_json['flags'] = {best_name: True, 'PCA': True}
        result_json['pca'] = histories
        return result_json

    n_jobs = args.jobs
    skipped = []

    if len(args.input) == 1:
        logger.info('Directory input')
        data_dir = Path(args.input[0])
        outdir = data_dir
        if args.output:
            outdir = Path(args.output)

        def do_dirin(fn: Path):
            logger.setLevel(os.environ.get('LOGLEVEL', 'INFO').upper())
            torch.set_num_threads(args.threads)
            file_id = fn.stem
            heatmaps, h = mhd.read(str(data_dir / f'{file_id}.mha'))
            with open(str(data_dir / f'{file_id}.json')) as f:
                original_json = json.load(f)
            try:
                result = fit(original_json, heatmaps)
            except Exception as e:
                return f"{file_id},{e}"
            result_json = result
            with open(outdir / f'{file_id}.json', 'w') as f:
                json.dump(result_json, f, indent=2)
            return None

        fns = sorted((data_dir).glob('*.json'))
        logger.info('Fit')
        skipped = Parallel(n_jobs=n_jobs)(delayed(do_dirin)(fn)
                                          for fn in tqdm(fns))
        if skipped: # remove None
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
        all_heatmaps, h = mhd.read(hm_filename)
        ndjson = []
        with open(ndjson_fn) as f:
            for line in f:
                ndjson.append(json.loads(line))

        def do(original_json,
               heatmaps):
            torch.set_num_threads(args.threads)
            file_id = original_json['filename']
            try:
                result = fit(original_json, heatmaps)
            except Exception as e:
                return f"{file_id},{e}"
            result_json = result
            return result_json

        logger.info('Fit')
        result_skipped = Parallel(n_jobs=n_jobs)(
            delayed(do)(*args)
            for args in tqdm(zip(ndjson, all_heatmaps), total=len(ndjson)))
        assert result_skipped is not None
        results = [r for r in result_skipped if isinstance(r, dict)]
        logger.info('Save ndjson')
        with open(args.output, 'w') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write('\n')
        logger.info('Done')
        skipped = [r for r in result_skipped if isinstance(r, str)]
    else:
        print('Invalid input')
        return 1

    assert skipped is not None
    if len(skipped) > 0:
        print(f'Save skipped {len(skipped)} files', args.skipped)
        with open(args.skipped, 'w') as f:
            f.write('\n'.join(skipped))
    else:
        print('No skipped files')
    return 0


if __name__ == '__main__':
    sys.exit(main())
