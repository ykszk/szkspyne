import argparse
import json
import sys
from typing import Tuple, Dict, Optional

import numpy as np
from logzero import logger

from szkmipy import mhd
from szkspyne import labelme_utils, point_utils, settings
from szkspyne.labelme_utils import LabelMeLine, ShapeDictLine
from szkspyne.utils import CompDecompGenerator
PROB_MAX = settings.PROB_MAX


def hm2labelme(
    pred,
    original_shape,
    image_filename: str,
    label_names,
    min_thresh: float,
    max_num_points: Optional[Dict[str, int]] = None,
):
    kps_per_ch = point_utils.local_center_of_masses(pred, min_thresh)
    for i_ch, (kps, label) in enumerate(zip(kps_per_ch, label_names)):
        max_num = max_num_points.get(label, len(kps)) if max_num_points else len(kps)
        if len(kps) > max_num:
            hm_values = [pred[i_ch, int(kp[0]), int(kp[1])] for kp in kps]
            kps = [kp for _, kp in sorted(zip(hm_values, kps), reverse=True, key=lambda t: t[0])[:max_num]]
            # resort kps by y
            kps = sorted(kps, key=lambda x: x[0])
            kps_per_ch[i_ch] = kps
    scale = original_shape[0] / pred.shape[1]
    kps_per_ch = [(np.array(kps, dtype=np.float32) * scale).tolist() for kps in kps_per_ch]
    json_data = labelme_utils.kps2labelme(kps_per_ch, image_filename, original_shape, label_names)
    return json_data


def save_kps_json(
    pred,
    original_shape,
    image_filename: str,
    json_filename: str,
    label_names,
    min_thresh: float,
    max_num_points: Optional[Dict[str, int]] = None,
):
    json_data = hm2labelme(pred, original_shape, image_filename, label_names, min_thresh, max_num_points)
    with open(json_filename, 'w') as f:
        f.write(json_data.model_dump_json())
    return json_data


def main():
    parser = argparse.ArgumentParser(description='Apply spine model.')

    parser.add_argument('heatmap', help='Input heatmap filename')
    parser.add_argument('ndjson', help='Input ndjson filename')
    parser.add_argument('output', help='Output directory')
    parser.add_argument('--save_heatmap', help='Save heatmap next to json', action='store_true')
    parser.add_argument('--save_ndjson', help='Save labelme data as an ndjson file')
    parser.add_argument('--jobs', help='The number of workers. Use all cores by default', metavar='<int>', type=int, default=-1)

    args = parser.parse_args()

    # lazy import
    import logzero
    from logzero import logger

    log_format = '%(color)s[%(levelname)1.1s %(asctime)s.%(msecs)d %(module)s:%(lineno)d]%(end_color)s %(message)s'
    formatter = logzero.LogFormatter(fmt=log_format)
    logzero.setup_default_logger(formatter=formatter)
    logger.info('Load modules')

    import json
    from pathlib import Path
    from typing import Union

    def heatmap2labelme(
        heatmap_filename: str,
        ndjson_filename: str,
        outdir: Union[str, Path],
        save_heatmap: bool,
        save_ndjson: Union[str, None],
        n_jobs: int,
    ):
        from joblib import Parallel, delayed

        outdir = Path(outdir)
        with open(ndjson_filename) as f:
            json_data = [ShapeDictLine.model_validate_json(l) for l in f.readlines()]

        def save_json(args: Tuple[np.ndarray, ShapeDictLine]) -> LabelMeLine:
            heatmap, data_line = args
            data = data_line.content
            heatmap = heatmap.transpose((2, 0, 1))  # from (Y,X,Ch) to (Ch,Y,X)
            original_shape = data.imageHeight, data.imageWidth
            filepath = Path(data_line.filename)
            stem = filepath.stem
            if save_heatmap:
                mhd.write(outdir / f'{stem}.mha', heatmap)
            min_thresh = 0.1 * PROB_MAX
            if save_ndjson is None:
                data = save_kps_json(
                    heatmap,
                    original_shape,
                    data.imagePath,
                    str(outdir / f'{stem}.json'),
                    settings.LABEL_NAMES,
                    min_thresh=min_thresh,
                    max_num_points=settings.MAX_NUM_POINTS,
                )
            else:
                data = hm2labelme(heatmap, original_shape, data.imagePath, settings.LABEL_NAMES, min_thresh=min_thresh, max_num_points=settings.MAX_NUM_POINTS)
            return data.into_labelme_line(data_line.filename)

        logger.info('start heatmap2labelme')
        # heatmaps, _ = mhd.read(heatmap_filename)
        # logger.info(f'heatmap shape: {heatmaps.shape}')
        # heatmaps = CompDecompGenerator.builder(method='zstandard', level=1)(heatmaps)
        heatmaps, _ = mhd.read_iterator(heatmap_filename)

        data = Parallel(n_jobs=n_jobs)(delayed(save_json)(arg) for arg in zip(heatmaps, json_data))
        logger.info('done heatmap2labelme')
        assert data is not None
        if save_ndjson:
            with open(save_ndjson, 'w') as f:
                for d in data:
                    f.write(d.model_dump_json())
                    f.write('\n')

    heatmap2labelme(args.heatmap, args.ndjson, args.output, args.save_heatmap, args.save_ndjson, args.jobs)


if __name__ == '__main__':
    sys.exit(main())
