import argparse
from pathlib import Path
from typing import List, NamedTuple, Union, Optional

import numpy as np
from logzero import logger
from sklearn.decomposition import PCA
from szkspyne import settings, labelme_utils

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

    def save(self, filename: Union[str, Path], model_name: str, labels: List[str], point_counts: List[int]):
        assert 2 * sum(point_counts) == self.mean_.shape[0]  # type: ignore
        np.savez(
            filename,
            model_name=model_name,
            ref_height=self.ref_height,
            mean=self.mean_,  # type: ignore
            components=self.components_,  # type: ignore
            explained_variance=self.explained_variance_,  # type: ignore
            explained_variance_ratio_=self.explained_variance_ratio_,  # type: ignore
            point_format='xy',
            labels=labels,
            point_counts=point_counts,
        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create PCA model(s) for 2D points.')
    parser.add_argument(
        'input',
        help='Input ndjson filename',
        metavar='<input>',
        nargs='?',
        default='../script/manual_annotation.ndjson',
    )
    parser.add_argument('output', help='Output directory', metavar='<output>', nargs='?', default='.')

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
        PointConfig('L6', 'ap_whole', VERT_LABELS, L6_POINT_COUNTS),
        PointConfig('L4', 'lateral_whole', VERT_LABELS, L4_POINT_COUNTS),
        PointConfig('L5', 'lateral_whole', VERT_LABELS, L5_POINT_COUNTS),
        PointConfig('L6', 'lateral_whole', VERT_LABELS, L6_POINT_COUNTS),
    ]
    if args.ext:
        CONFIGS.extend(
            [
                PointConfig('L4-EXT', 'ap_whole', LABELS_EXT, L4_POINT_COUNTS + [2] * 5),
                PointConfig('L5-EXT', 'ap_whole', LABELS_EXT, L5_POINT_COUNTS + [2] * 5),
                PointConfig('L6-EXT', 'ap_whole', LABELS_EXT, L6_POINT_COUNTS + [2] * 5),
            ]
        )
    all_data = labelme_utils.ShapeDictLine.from_ndjson(args.input)
    logger.info('Original data: %d', len(all_data))

    for config in CONFIGS:
        image_type = config.image_type
        logger.info('%s %s', config.name, image_type)
        valid_data: List[labelme_utils.ShapeDictLine] = []
        for data_line in all_data:
            d = data_line.content
            if image_type not in d.flags:
                continue
            ns = {k: len(v) for k, v in d.shapes['point'].items()}
            valid = True
            for l, n in zip(config.labels, config.point_counts):
                if l not in ns or ns[l] != n:
                    valid = False
                    break
            if valid:
                valid_data.append(data_line)
        logger.info('%d %d', len(all_data), len(valid_data))

        pca = PointsPCA(REF_HEIGHT)

        vecs = []
        for vd_line in valid_data:
            vd = vd_line.content
            point_data = vd.shapes['point']
            c7_tl = point_data['TL'][0]
            s_tl = point_data['TL'][-1]
            pts = []
            for l in config.labels:
                pts.append(pca.to_ref_space(c7_tl, s_tl, point_data[l]))
            pts = np.concatenate(pts).ravel()
            vecs.append(pts)
        vecs = np.stack(vecs)

        pca.fit(vecs)
        npz_filename = outdir / f'pca_{image_type}_{config.name}.npz'
        pca.save(npz_filename, config.name, config.labels, config.point_counts)
