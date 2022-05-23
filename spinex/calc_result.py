#%%
import os
import pandas as pd
import numpy as np
import mhd
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage.measurements import center_of_mass
from skimage import measure
import cv2
import itertools
import argparse
from joblib import Parallel, delayed
from logzero import logger


def extract_center_of_masses(heatmaps):
    '''
    Center of mass of pixels with the maximum value

    Examples:

    >>> a = np.zeros((2, 3, 3))
    >>> a[0, 1, 1] = 1
    >>> a[1, 1, 2] = 2
    >>> a[1, 1, 1] = 1
    >>> extract_center_of_masses(a)
    [(1.0, 1.0), (1.0, 2.0)]
    '''
    keypoints = []
    for ch in heatmaps:
        max_value = np.max(ch)
        if max_value <= 0:
            raise RuntimeError('Empty heatmap')
        center = center_of_mass(ch == max_value)
        keypoints.append(center)
    return keypoints


def local_center_of_masses(heatmaps):
    '''
    Examples:

    >>> a = np.zeros((1, 5, 5))
    >>> a[0, 1, 1] = 1
    >>> a[0, 3, 4] = 1
    >>> local_center_of_masses(a)
    [[(1.0, 1.0), (3.0, 4.0)]]
    '''
    local_centers_per_ch = []
    for ch in heatmaps:
        max_value = np.max(ch)
        if max_value <= 0:
            raise RuntimeError('Empty heatmap')
        thresh = max_value / 2
        all_labels = measure.label(ch >= thresh)
        local_centers = []
        for i in range(1, np.max(all_labels) + 1):
            cc = (all_labels == i)
            center = center_of_mass(ch * (cc == cc.max()))
            local_centers.append(center)
        local_centers_per_ch.append(local_centers)
    return local_centers_per_ch


def h2p_affinity(heatmap):
    all_candidates = local_center_of_masses(heatmap)

    keypoints = []
    best_scores = []
    for i, ch in enumerate(heatmap[4:]):
        c1s, c2s = all_candidates[2 * i], all_candidates[2 * i + 1]
        if len(c1s) == 0 and len(c2s) == 0:
            pair = [c1s[0], c2s[0]]
        else:
            scores, pairs = [], []
            for c1, c2 in itertools.product(c1s, c2s):
                line = np.zeros_like(ch)
                ic1, ic2 = tuple(np.round(c1).astype(int)), tuple(
                    np.round(c2).astype(int))
                line = cv2.line(line, ic2, ic1, 1, 1)
                line_sum = np.sum(ch * line) / np.sum(line)
                scores.append(line_sum)
                pairs.append((c1, c2))
            best_scores.append(np.max(scores))
            pair = pairs[np.argmax(scores)]
        keypoints.extend(pair)
    minmax = []
    for hm in heatmap[:4]:
        minmax.append((hm.min()))
        minmax.append((hm.max()))
    return keypoints, minmax, best_scores


def main():
    parser = argparse.ArgumentParser(description='Calculate anatomical point.')
    parser.add_argument(
        '--result',
        help='Result directory. default: %(default)s',
        metavar='<dir>',
        default=
        '/poseidon/users/suzuki/projects/scoliosis/result/pre_spineweb_verse')
    # default='../result')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    resultdir = Path(args.result)
    df = pd.read_csv(resultdir / 'dataset.csv')
    point_cols = [
        c for c in df.columns if (c.endswith(' x') or c.endswith(' y'))
    ]
    PRED_PREFIX = 'pred '
    for c in point_cols:
        df[PRED_PREFIX + c] = None
    df['slice_pos'] = 0

    logger.info('start calc keypoints')
    for set_id, df_set in df.groupby('set_id'):
        if set_id != 0:
            logger.info('skip %d', set_id)
            continue
        y_preds, _ = mhd.read(resultdir / '{}/y_pred.mha'.format(set_id))
        assert y_preds.shape[-1] == len(point_cols) // 2
        y_preds = np.moveaxis(y_preds, -1, 1)  # to channel first
        keypoints_per_ch_per_img = Parallel(n_jobs=os.cpu_count())(
            delayed(local_center_of_masses)(y_pred)
            for y_pred in tqdm(y_preds, desc='set {}'.format(set_id)))
        for i, keypoints_per_ch in enumerate(keypoints_per_ch_per_img):
            for ch, kps in enumerate(keypoints_per_ch):
                kps = np.array(kps)
                df.at[df_set.index[i],
                      PRED_PREFIX + point_cols[ch]] = tuple(kps[:, 0])
                df.at[df_set.index[i],
                      PRED_PREFIX + point_cols[ch + 1]] = tuple(kps[:, 1])
        df.loc[df_set.index, 'slice_pos'] = list(range(len(df_set)))
        break
    logger.info('end calc landmarks')
    df.to_csv(resultdir / 'result.csv', index=False)


if __name__ == "__main__":
    import sys
    sys.exit(main())