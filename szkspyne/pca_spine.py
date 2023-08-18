#%%
import json
from pathlib import Path
from typing import List, NamedTuple, Union
import copy

import numpy as np
from logzero import logger
from sklearn.decomposition import PCA

from szkspyne import settings


class PointConfig(NamedTuple):
    name: str
    image_type: str
    labels: List[str]
    point_counts: List[int]


class PointsPCA(PCA):

    def __init__(self, ref_height: float):
        super().__init__()
        self.ref_height = ref_height

    def to_ref_mat(self, top, bottom):
        '''
        Return transformation matrix that:
        1) Scales points so that |top - bottom| becomes 1
        2) Rotates points so that (top - bottom) becomes vertical
        3) Translates points so that top becomes the origin
        '''
        top, bottom = np.array(top), np.array(bottom)
        v = bottom - top  # type: ignore
        s = self.ref_height / np.linalg.norm(v)
        s = np.diag([s, s, 1.0])
        cos = v[1] / np.linalg.norm(v)
        sin = np.sign(v[0]) * np.sqrt(1 - cos**2)
        rot = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
        t = np.array([[1, 0, -top[0]], [0, 1, -top[1]], [0, 0, 1]])
        mat = s @ rot @ t
        return mat

    def to_target_mat(self, top, bottom):
        mat = self.to_ref_mat(top, bottom)
        return np.linalg.inv(mat)

    def to_ref_space(self, top, bottom, points):
        mat = self.to_ref_mat(top, bottom)
        padded = np.pad(points, ((0, 0), (0, 1)), constant_values=1)
        return (mat @ padded.T).T[:, :2]

    def to_target_space(self, top, bottom, points):
        mat = self.to_target_mat(top, bottom)
        padded = np.pad(points, ((0, 0), (0, 1)), constant_values=1)
        return (mat @ padded.T).T[:, :2]

    def save(self, filename: Union[str, Path], model_name: str,
             labels: List[str], point_counts: List[int]):
        assert 2 * sum(point_counts) == self.mean_.shape[0]  # type: ignore
        np.savez(
            filename,
            model_name=model_name,
            ref_height=self.ref_height,
            mean=self.mean_,  # type: ignore
            components=self.components_,  # type: ignore
            explained_variance=self.explained_variance_,  # type: ignore
            explained_variance_ratio_=self.
            explained_variance_ratio_,  # type: ignore
            point_format='xy',
            labels=labels,
            point_counts=point_counts)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Create PCA model(s) for 2D points.')
    parser.add_argument('input',
                        help='Input ndjson/csv filename',
                        metavar='<input>',
                        nargs='?',
                        default='../script/manual_annotation.ndjson')
    parser.add_argument('output',
                        help='Output directory',
                        metavar='<output>',
                        nargs='?',
                        default='.')
    parser.add_argument('--mirror',
                        help='Add horizontally mirrored point set to the dataset',
                        action='store_true')

    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning('Unknown arguments:%s', unknown)
    outdir = Path(args.output)
    REF_HEIGHT = 1000
    VERT_LABELS = settings.VERTEBRA_LABEL_NAMES
    L5_POINT_COUNTS = [19, 19, 18, 18]
    L4_POINT_COUNTS = (np.array(L5_POINT_COUNTS) - 1).tolist()
    L6_POINT_COUNTS = (np.array(L5_POINT_COUNTS) + 1).tolist()
    LABELS_EXT = VERT_LABELS + settings.NON_VERTEBRA_LABEL_NAMES
    CONFIGS = [
        PointConfig('L4', 'ap_whole', VERT_LABELS, L4_POINT_COUNTS),
        PointConfig('L5', 'ap_whole', VERT_LABELS, L5_POINT_COUNTS),
        PointConfig('L4', 'lateral_whole', VERT_LABELS, L4_POINT_COUNTS),
        PointConfig('L5', 'lateral_whole', VERT_LABELS, L5_POINT_COUNTS),
        PointConfig('L4-EXT', 'ap_whole', LABELS_EXT,
                    L4_POINT_COUNTS + [2] * 5),
        PointConfig('L5-EXT', 'ap_whole', LABELS_EXT,
                    L5_POINT_COUNTS + [2] * 5),
        # PointConfig('L6', LABELS, L6_POINT_COUNTS, 'lateral_whole'),
    ]
    with open(args.input) as f:
        all_data = [json.loads(line) for line in f]
    logger.info('Original data: %d', len(all_data))
    if args.mirror:
        mirrored = copy.deepcopy(all_data)
        for d in mirrored:
            width = d['imageWidth']
            for point in d['point'].values():
                for p in point:
                    p[0][0] = width - p[0][0]
        all_data += mirrored
        

    # preprocess
    for d in all_data:
        ns = {k: len(v) for k, v in d['point'].items()}
        np_pts = {}
        for label, pts in d['point'].items():
            if label in VERT_LABELS:
                pts.sort(key=lambda v: v[0][1])  # sort by y
            else:
                pts.sort(key=lambda v: v[0][0])  # sort by x
            np_pts[label] = np.array(pts)[:, 0]  # squeeze (N, 1, 2) -> (N, 2)
        d['point'] = np_pts
    # add potential pseudo L4
    l4_data = copy.deepcopy(all_data)
    for d in l4_data:
        np_pts = {}
        for label, pts in d['point'].items():
            if label in VERT_LABELS:
                np_pts[label] = pts[1:].copy()
            else:
                np_pts[label] = pts.copy()
        d['point'] = np_pts
    all_data += l4_data

    for config in CONFIGS:
        image_type = config.image_type
        logger.info('%s %s', config.name, image_type)
        valid_data = []
        for d in all_data:
            if d['image_type'] != image_type:
                continue
            ns = {k: len(v) for k, v in d['point'].items()}
            valid = True
            for l, n in zip(config.labels, config.point_counts):
                if l not in ns or ns[l] != n:
                    valid = False
                    break
            if valid:
                valid_data.append(d)
        logger.info('%d %d', len(all_data), len(valid_data))

        pca = PointsPCA(REF_HEIGHT)

        vecs = []
        for vd in valid_data:
            c7_tl = vd['point']['TL'][0]
            s_tl = vd['point']['TL'][-1]
            pts = []
            for l in config.labels:
                pts.append(pca.to_ref_space(c7_tl, s_tl, vd['point'][l]))
            pts = np.concatenate(pts).ravel()
            vecs.append(pts)
        vecs = np.stack(vecs)

        pca.fit(vecs)
        npz_filename = outdir / f'pca_{image_type}_{config.name}.npz'
        pca.save(npz_filename, config.name, config.labels, config.point_counts)
