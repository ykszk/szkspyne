#%%
import json
from re import match
import numpy as np
from sklearn.decomposition import PCA
from logzero import logger
from pathlib import Path
from typing import NamedTuple, List, Union


class PointConfig(NamedTuple):
    name: str
    labels: List[str]
    point_counts: List[int]


class PointsPCA(PCA):

    def __init__(self, ref_height: float):
        super().__init__()
        self.ref_height = ref_height

    def to_ref_mat(self, top, bottom):
        top, bottom = np.array(top), np.array(bottom)
        v = bottom - top
        s = self.ref_height / np.linalg.norm(v)
        cos = v[1] / np.linalg.norm(v)
        sin = np.sign(v[0]) * np.sqrt(1 - cos**2)
        rot = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
        t = np.array([[1, 0, -top[0]], [0, 1, -top[1]], [0, 0, 1]])
        mat = (s * np.eye(3)) @ rot @ t
        return mat

    def to_target_mat(self, top, bottom):
        mat = self.to_ref_mat(top, bottom)
        return np.linalg.inv(mat)

    def to_ref_space(self, top, bottom, points):
        mat = self.to_ref_mat(top, bottom)
        padded = np.pad(points, ((0,0),(0,1)), mode='constant', constant_values=1)
        return (mat @ padded.T).T[:, :2]

    def to_target_space(self, top, bottom, points):
        mat = self.to_target_mat(top, bottom)
        padded = np.pad(points, ((0,0),(0,1)), mode='constant', constant_values=1)
        return (mat @ padded.T).T[:, :2]

    def save(self, filename: Union[str, Path], point_counts: List[int]):
        assert 2 * sum(point_counts) == self.mean_.shape[0]
        np.savez(
            filename,
            ref_height=self.ref_height,
            mean=self.mean_,  # type: ignore
            components=self.components_,  # type: ignore
            explained_variance=self.explained_variance_,  # type: ignore
            explained_variance_ratio_=self.
            explained_variance_ratio_,  # type: ignore
            point_format='xy',
            labels=LABELS,
            point_counts=point_counts)
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filename = '../script/manual_annotation.ndjson'
    REF_HEIGHT = 1000
    LABELS = ['TL', 'TR', 'BL', 'BR']
    L5_POINT_COUNTS = [19, 19, 18, 18]
    L4_POINT_COUNTS = (np.array(L5_POINT_COUNTS) - 1).tolist()
    L6_POINT_COUNTS = (np.array(L5_POINT_COUNTS) + 1).tolist()
    CONFIGS = [
        PointConfig('L4', LABELS, L4_POINT_COUNTS),
        PointConfig('L5', LABELS, L5_POINT_COUNTS),
        # PointConfig('L6', LABELS, L6_POINT_COUNTS, 'lateral_whole'),
    ]
    import itertools
    for config, image_type in itertools.product(CONFIGS,
                                                ('ap_whole', 'lateral_whole')):
        logger.info('%s %s', config.name, image_type)
        with open(filename) as f:
            data = [json.loads(line) for line in f]
        valid_data = []
        for d in data:
            if d['image_type'] != image_type:
                continue
            ns = {k: len(v) for k, v in d['point'].items()}
            np_pts = {}
            for label, pts in d['point'].items():
                pts.sort(key=lambda v: v[0][1])  # sort by y
                np_pts[label] = np.array(pts)[:, 0]
            d['point'] = np_pts
            valid = True
            for l, n in zip(config.labels, config.point_counts):
                if ns[l] != n:
                    valid = False
                    break
            if valid:
                valid_data.append(d)
        logger.info('%d %d', len(data), len(valid_data))

        pca = PointsPCA(REF_HEIGHT)
        for vd in valid_data:
            tl = vd['point']['TL'][0]
            bl = vd['point']['TL'][-1]
            for l in ['TL', 'TR', 'BL', 'BR']:
                vd['point'][l] = pca.to_ref_space(tl, bl, vd['point'][l])
        # for vd in valid_data[:5]:
        #     pts = vd['point']['TL']
        #     plt.scatter(pts[:, 0], pts[:, 1], alpha=.2)

        vecs = []
        for vd in valid_data:
            pts = []
            for l in ['TL', 'TR', 'BL', 'BR']:
                pts.append(vd['point'][l])
            pts = np.concatenate(pts).ravel()
            vecs.append(pts)
        vecs = np.stack(vecs)

        pca.fit(vecs)
        npz_filename = f'pca_{image_type}_{config.name}.npz'
        pca.save(npz_filename, config.point_counts)
