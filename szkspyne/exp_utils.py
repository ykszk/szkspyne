import json
import sys
from pathlib import Path
from typing import List, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from logzero import logger
from PIL import Image
from tqdm import tqdm

from szkspyne import img_utils, transforms
from szkspyne.settings import LABEL_NAMES


def plot_history(df):
    melted = pd.melt(df,
                     value_vars=['train_loss', 'val_loss'],
                     id_vars=['step'],
                     value_name='loss',
                     var_name='subset')
    sns.lineplot(data=melted, x='step', y='loss', hue='subset')


def load_data(
        row, data_root: Path,
        label_names: Sequence[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
    '''Load data from a row. Keypoints should be np.ndarray
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


def load_dataset(df: pd.DataFrame, height: Union[int, None], data_root: Path,
                 label_names: Sequence[str], num_workers: int,
                 down_factor: int, clahe: bool):
    '''
    load images and keypoints
    '''

    def load(row):
        x, kp = load_data(row, data_root, label_names)
        assert kp[0].size != 0
        if height is not None:
            x, kp = img_utils.resize_height(height, x, kp)
        if clahe:
            x = transforms.apply_clahe(x)
        return x, kp

    data = Parallel(n_jobs=num_workers)(
        delayed(load)(row) for _, row in tqdm(df.iterrows(), total=len(df)))
    xs, kps = zip(*data)

    shapes = [x.shape for x in xs]
    max_shape = np.max(shapes, axis=0)

    new_shape = (down_factor *
                 np.ceil(np.array(max_shape) / down_factor)).astype(
                     np.int64).tolist()
    xs = [img_utils.pad_to_shape(x, new_shape, 0) for x in xs]
    return xs, kps


def ndjson2csv(filename: Union[str, Path],
               exclude_flags: Set[str]) -> pd.DataFrame:
    with open(filename) as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        data = json.loads(line)
        flags = set(data['flags'])
        intxn = flags.intersection(exclude_flags)
        if len(intxn) > 0:
            print('Skip', data['filepath'], intxn)
            continue
        if 'circle' in data:  # TODO: This is temporary. Circles should be handled as circles in the future!
            data['point'].update(data['circle'])
        points = {
            k: np.array(p)[:, 0]
            for k, p in data['point'].items() if k in LABEL_NAMES
        }
        for name in LABEL_NAMES:
            if name not in points.keys():
                points[name] = np.array(())
        row = {'filepath': data['filepath'], 'flags': data['flags'], 'group_id': data.get('group_id', None)}
        row.update(points)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def main():
    import argparse

    from logzero import logger
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
            df = ndjson2csv(csv_filename, set())
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
            df = ndjson2csv(csv_filename, set())
        else:
            df = pd.read_csv(csv_filename)
        import settings
        import transforms
        from img_utils import resize_height
        from point_utils import flatten_nest, partition
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
    p_loader.add_argument('--transform',
                          help='Transformation: %(default)s',
                          choices=['random', 'base'],
                          default='random')
    p_loader.add_argument('--output',
                          help='Optional output image. e.g. kp_dataset.svg')
    p_loader.add_argument(
        '--mhd_x',
        help='Optional output mhd image for training input. e.g. kp_dataset.mhd'
    )
    p_loader.add_argument(
        '--mhd_y',
        help=
        'Optional output mhd image for training output. e.g. kp_dataset.mhd')

    def command_loader(args):
        logger.info('Loader')
        csv_filename, data_root, label_names = args.csv, args.data_root, args.label_names
        if csv_filename.endswith('.ndjson'):
            df = ndjson2csv(csv_filename, set())
        else:
            df = pd.read_csv(csv_filename)
        import settings
        import transforms
        from img_utils import pad_to_shape, resize_height
        from point_utils import KeypointDataset, flatten_nest, partition
        from torch.utils.data import DataLoader

        flip_pairs = settings.FLIP_PAIRS
        down_factor = 256
        xs, kps = load_dataset(df.sample(n=args.rows), args.height,
                               Path(data_root), label_names, 4, down_factor,
                               True)

        tf = transforms.random_transform if args.transform == 'random' else transforms.base_kps_transform
        dataset = KeypointDataset(xs,
                                  kps,
                                  None,
                                  flip_pairs,
                                  tf,
                                  sigma=args.sigma)
        loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.rows,
            num_workers=4,
        )
        plt.figure(figsize=(10, 10))
        for batch_xs, batch_ys in loader:
            for i_subplot, (x, y) in enumerate(zip(batch_xs, batch_ys)):
                plt.subplot(args.rows,
                            len(y) + 1,
                            i_subplot * (len(y) + 1) + 1)
                plt.imshow(x.squeeze(), 'gray')
                plt.axis('off')
                for i, hm in enumerate(y):
                    u8img = np.round(np.clip(hm.numpy(), 0, 1) * 255).astype(
                        np.uint8)
                    plt.subplot(args.rows,
                                len(y) + 1,
                                i_subplot * (len(y) + 1) + i + 2)
                    plt.imshow(u8img, 'gray')
                    plt.axis('off')
            if args.mhd_x:
                from szkmipy import mhd
                logger.info('Save x batch')
                print('batch shape in utils', batch_xs.shape)
                batch_xs = batch_xs.squeeze()  # remove channel dim
                mhd.write(args.mhd_x, batch_xs.cpu().numpy())
            if args.mhd_y:
                from szkmipy import mhd
                logger.info('Save y batch')
                batch_ys = batch_ys.reshape(
                    (-1, batch_ys.shape[2],
                     batch_ys.shape[3]))  # merge channel and batch dims
                mhd.write(args.mhd_y, batch_ys.cpu().numpy())
        if args.output:
            plt.gcf().savefig(args.output,
                              transparent=True,
                              bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        return 0

    p_loader.set_defaults(handler=command_loader)

    args = parser.parse_args()
    if not hasattr(args, 'handler'):
        print('No command was specified.')
        parser.print_help()
        sys.exit(1)
    return (args.handler(args))


if __name__ == '__main__':
    sys.exit(main())