import sys
import json
from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Tuple, List
from pathlib import Path
from PIL import Image

from settings import LABEL_NAMES


def plot_history(df, set_ylim=True):
    df = pd.DataFrame((df['train_loss'].dropna().reset_index(drop=True),
                       df['val_loss'].dropna().reset_index(drop=True))).T
    df.plot(y=['train_loss', 'val_loss'])
    if set_ylim and len(df) > 1:
        plt.ylim(df[['train_loss', 'val_loss']].min().min(),
                 sorted(df['train_loss'])[-2])


def load_data(
        row, data_root: Path,
        label_names: Sequence[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
    '''Load data from a row.
    '''
    img_path = data_root / row.filepath
    x = np.array(Image.open(img_path))
    kps = [row.get(l) for l in label_names]
    # flip kps to change format from (x, y) to (y, x)
    kps = [
        np.flip(kp, axis=-1) if isinstance(kp, np.ndarray) else np.array(())
        for kp in kps
    ]
    return x, kps


def _load_data(
        row, data_root: Path,
        label_names: Sequence[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
    '''Load data from a row
    '''
    # img_path = (mha_root / row.filepath).with_suffix('.mha')
    # x, _ = mhd.read(img_path)
    img_path = data_root / row.filepath
    x = np.array(Image.open(img_path))
    # assert x.dtype == np.uint16, 'np.uint16 is expected'
    kp = []
    for l in label_names:
        px, py = row.get(l + ' x'), row.get(l + ' y')
        px, py = eval(px), eval(py)
        pts = np.array(((py, px)))
        kp.append(pts.T)
    # kp = np.array(kp)
    return x, kp


def ndjson2csv(filename: Union[str, Path]) -> pd.DataFrame:
    with open(filename) as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        data = json.loads(line)
        if '側面右向き' in data['flags']:
            print('Skip right facing', data['filepath'])
        if '除外' in data['flags']:
            print('Skip 除外', data['filepath'])
        if 'circle' in data:  # TODO: This is temporary. Circles should be handled properly in the future!
            data['point'].update(data['circle'])
        points = {
            k: np.array(p)[:, 0]
            for k, p in data['point'].items() if k in LABEL_NAMES
        }
        for name in LABEL_NAMES:
            if name not in points.keys():
                points[name] = np.array(())
        row = {'filepath': data['filepath']}
        row.update(points)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def main():
    from logzero import logger
    import argparse
    parser = argparse.ArgumentParser(
        description='Try utility functions for debugging.')
    parser.add_argument('csv',
                        help='CSV or NDJSON filename: %(default)s',
                        nargs='?',
                        default='../script/verse.csv')
    parser.add_argument('data_root',
                        help='Data root directory: %(default)s',
                        nargs='?',
                        default='../../data')
    parser.add_argument('--label_names',
                        help='Label names: %(default)s',
                        nargs='?',
                        default=LABEL_NAMES)
    parser.add_argument('--height',
                        help='Resized image height: %(default)d',
                        default=1024)

    subparsers = parser.add_subparsers()
    p_summary = subparsers.add_parser(
        'load_data', help="load a set of image and keypoints from a csv file.")
    p_summary.add_argument('--output',
                           help='Optional output image. e.g. keypoints.svg')

    def command_load_data(args):
        csv_filename, data_root, label_names = args.csv, args.data_root, args.label_names
        if csv_filename.endswith('.ndjson'):
            df = ndjson2csv(csv_filename)
        else:
            df = pd.read_csv(csv_filename)
        from img_utils import resize_height
        for _, row in df.iterrows():
            x, kps = load_data(row, Path(data_root), label_names)
            x, kps = resize_height(args.height, x, kps)
            print('Image shape:', x.shape)
            print('Points shape:', [kp.shape for kp in kps])
            plt.imshow(x, 'gray')
            for pts in kps:
                if len(pts) > 0:
                    plt.scatter(pts[:, 1], pts[:, 0], s=2)
            plt.axis('off')
            if args.output:
                plt.gcf().savefig(args.output,
                                  transparent=True,
                                  bbox_inches='tight')
            else:
                plt.show()
            plt.close()
            break
        return 0

    p_summary.set_defaults(handler=command_load_data)

    p_transform = subparsers.add_parser(
        'transform', help="Test transforms.random_transform.")
    p_transform.add_argument('--rows',
                             help='The number of rample data: %(default)d',
                             type=int,
                             default=5)
    p_transform.add_argument(
        '--columns',
        help='The number of random transformations: %(default)d',
        type=int,
        default=5)
    p_transform.add_argument(
        '--output', help='Optional output image. e.g. random_transforms.svg')

    def command_transform(args):
        logger.info('Transform')
        csv_filename, data_root, label_names = args.csv, args.data_root, args.label_names
        if csv_filename.endswith('.ndjson'):
            df = ndjson2csv(csv_filename)
        else:
            df = pd.read_csv(csv_filename)
        from img_utils import resize_height
        from point_utils import flatten_nest, partition
        import transforms
        import settings
        flip_pairs = settings.FLIP_PAIRS
        plt.figure(figsize=(7, 7))
        i_subplot = 1
        for idx, row in df.sample(n=args.rows).iterrows():
            logger.info('%s', idx)
            x, kps = load_data(row, Path(data_root), label_names)
            x, kps = resize_height(args.height, x, kps)
            for _ in range(args.columns):
                tx, tkps = transforms.random_transform(x, kps, flip_pairs)
                tx = tx.squeeze()
                tkps = [np.array(tkp) for tkp in tkps]
                plt.subplot(args.rows, args.columns, i_subplot)
                i_subplot += 1
                plt.imshow(tx, 'gray')
                for pts in tkps:
                    if len(pts) > 0:
                        plt.scatter(pts[:, 1], pts[:, 0], s=2)
                    else:
                        next(plt.gca()._get_lines.prop_cycler)
                plt.axis('off')
        if args.output:
            plt.gcf().savefig(args.output,
                              transparent=True,
                              bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        return 0

    p_transform.set_defaults(handler=command_transform)

    p_loader = subparsers.add_parser('loader', help="Test KeypointDataset.")
    p_loader.add_argument('--rows',
                          help='The number of rample data: %(default)d',
                          type=int,
                          default=5)
    p_loader.add_argument('--sigma',
                          help='Sigma for heatmaps: %(default)d',
                          type=int,
                          default=5)
    p_loader.add_argument('--output',
                          help='Optional output image. e.g. kp_dataset.svg')

    def command_loader(args):
        logger.info('Loader')
        csv_filename, data_root, label_names = args.csv, args.data_root, args.label_names
        if csv_filename.endswith('.ndjson'):
            df = ndjson2csv(csv_filename)
        else:
            df = pd.read_csv(csv_filename)
        from img_utils import resize_height, pad_to_shape
        from point_utils import flatten_nest, partition, KeypointDataset
        import transforms
        import settings
        from torch.utils.data import DataLoader

        flip_pairs = settings.FLIP_PAIRS
        plt.figure(figsize=(7, 7))
        xs, kps = [], []
        for idx, row in df.sample(n=args.rows).iterrows():
            logger.info('%s', idx)
            x, kp = load_data(row, Path(data_root), label_names)
            x, kp = resize_height(args.height, x, kp)
            xs.append(x), kps.append(kp)
        shapes = [x.shape for x in xs]
        max_shape = np.max(shapes, axis=0)
        down_factor = 256
        new_shape = (down_factor *
                        np.ceil(np.array(max_shape) / down_factor)).astype(
                            np.int64).tolist()
        logger.info('Pad to %s', new_shape)
        xs = [pad_to_shape(x, new_shape, 0) for x in xs]
        dataset = KeypointDataset(xs,
                                  kps,
                                  None,
                                  flip_pairs,
                                  transforms.random_transform,
                                  sigma=args.sigma)
        loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.rows,
            num_workers=4,
        )
        for batch_xs, batch_ys in loader:
            for i_subplot, (x, y) in enumerate(zip(batch_xs, batch_ys)):
                # import mhd
                # mhd.write(f'{i_subplot}.mha', y.numpy())
                plt.subplot(args.rows, len(y)+1, i_subplot*(len(y)+1)+1)
                plt.imshow(x.squeeze(), 'gray')
                plt.axis('off')
                for i, hm in enumerate(y):
                    u8img = np.round(np.clip(hm.numpy(), 0, 1)*255).astype(np.uint8)
                    plt.subplot(args.rows, len(y)+1, i_subplot*(len(y)+1)+i+2)
                    plt.imshow(u8img, 'gray')
                    plt.axis('off')
        if args.output:
            plt.gcf().savefig(args.output,
                              transparent=True,
                              bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        return 0

    p_loader.set_defaults(handler=command_loader)

    # parser.add_argument('-s',
    #                     '--switch',
    #                     help='Switch argument',
    #                     action='store_true')

    args = parser.parse_args()
    if not hasattr(args, 'handler'):
        print('No command was specified.')
        parser.print_help()
        sys.exit(1)
    return (args.handler(args))


if __name__ == '__main__':
    sys.exit(main())