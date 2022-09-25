#%%
import json
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

    def to_ref_space(self, top, bottom, points):
        height = bottom[1] - top[1]
        s = self.ref_height / height
        t = top
        return s * (points - t)

    def to_target_space(self, top, bottom, points):
        height = bottom[1] - top[1]
        s = self.ref_height / height
        t = top
        return points / s + t

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
