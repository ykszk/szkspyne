import argparse
import sys


# Cannot lazy load for pickling
import json

import labelme_utils
import numpy as np
import point_utils
import settings
from szkmipy import mhd

PROB_MAX = settings.PROB_MAX


def hm2labelme(pred,
               original_shape,
               image_filename: str,
               label_names,
               min_thresh=.25):
    kps_per_ch = point_utils.local_center_of_masses(pred, min_thresh)
    scale = original_shape[0] / pred.shape[1]
    kps_per_ch = [(np.array(kps, dtype=np.float32) * scale).tolist()
                  for kps in kps_per_ch]
    json_data = labelme_utils.kps2labelme(kps_per_ch, image_filename,
                                          original_shape, label_names)
    return json_data


def save_kps_json(pred,
                  original_shape,
                  image_filename: str,
                  json_filename: str,
                  label_names,
                  min_thresh=.25):
    json_data = hm2labelme(pred, original_shape, image_filename, label_names,
                           min_thresh)
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    return json_data



def main():
    parser = argparse.ArgumentParser(description='Apply spine model.')

    parser.add_argument('heatmap', help='Input heatmap filename')
    parser.add_argument('ndjson', help='Input ndjson filename')
    parser.add_argument('output', help='Output directory')
    parser.add_argument('--save_heatmap',
                        help='Save heatmap next to json',
                        action='store_true')
    parser.add_argument('--save_ndjson',
                        help='Save labelme data as an ndjson file')
    parser.add_argument('--jobs',
                        help='The number of workers: %(default)s',
                        metavar='<int>',
                        type=int,
                        default=4)

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

    def heatmap2labelme(heatmap_filename: str, ndjson_filename: str,
                        outdir: Union[str, Path], save_heatmap: bool,
                        save_ndjson: Union[str, None], n_jobs: int):
        from joblib import Parallel, delayed
        outdir = Path(outdir)
        with open(ndjson_filename) as f:
            json_data = [json.loads(l) for l in f.readlines()]

        def save_json(args):
            heatmap, data = args
            heatmap = heatmap.transpose(
                (2, 0, 1))  # from (Y,X,Ch) to (Ch,Y,X)
            original_shape = data['imageHeight'], data['imageWidth']
            filepath = Path(data['filepath'])
            stem = filepath.stem
            if save_heatmap:
                mhd.write(outdir / f'{stem}.mha', heatmap)
            if save_ndjson is None:
                data = save_kps_json(heatmap,
                                     original_shape,
                                     f'{stem}.jpg',
                                     str(outdir / f'{stem}.json'),
                                     settings.LABEL_NAMES,
                                     min_thresh=.1 * PROB_MAX)
            else:
                data = hm2labelme(heatmap,
                                  original_shape,
                                  f'{stem}.jpg',
                                  settings.LABEL_NAMES,
                                  min_thresh=.1 * PROB_MAX)
            data['filename'] = stem
            return data

        logger.info('start heatmap2labelme')
        # heatmaps, _ = mhd.read_iterator(heatmap_filename)
        heatmaps, _ = mhd.read(heatmap_filename)
        data = Parallel(n_jobs=n_jobs)(delayed(save_json)(arg)
                                       for arg in zip(heatmaps, json_data))
        logger.info('done heatmap2labelme')
        assert data is not None
        if save_ndjson:
            with open(save_ndjson, 'w') as f:
                for d in data:
                    f.write(json.dumps(d, ensure_ascii=False))
                    f.write('\n')

    heatmap2labelme(args.heatmap, args.ndjson, args.output, args.save_heatmap,
                    args.save_ndjson, args.jobs)


if __name__ == '__main__':
    sys.exit(main())
