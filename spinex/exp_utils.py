import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Tuple
from pathlib import Path
from PIL import Image

LABEL_NAMES = ['TL', 'TR', 'BL', 'BR']


def plot_history(df, set_ylim=True):
    df = pd.DataFrame((df['train_loss'].dropna().reset_index(drop=True),
                       df['val_loss'].dropna().reset_index(drop=True))).T
    df.plot(y=['train_loss', 'val_loss'])
    if set_ylim and len(df) > 1:
        plt.ylim(df[['train_loss', 'val_loss']].min().min(),
                 sorted(df['train_loss'])[-2])


def load_data(row, data_root: Path,
              label_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
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
    kp = np.array(kp)
    return x, kp


def main():
    from logzero import logger
    import argparse
    parser = argparse.ArgumentParser(
        description='Try utility functions for debugging.')
    parser.add_argument('csv',
                        help='CSV filename: %(default)s',
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
        df = pd.read_csv(csv_filename)
        from img_utils import resize_height
        for _, row in df.iterrows():
            x, kps = load_data(row, Path(data_root), label_names)
            x, kps = resize_height(args.height, x, kps)
            print('Image shape:', x.shape)
            print('Points shape:', kps.shape)
            plt.imshow(x, 'gray')
            for pts in kps:
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
                             default=5)
    p_transform.add_argument(
        '--columns',
        help='The number of random transformations: %(default)d',
        default=5)
    p_transform.add_argument(
        '--output', help='Optional output image. e.g. random_transforms.svg')

    def command_transform(args):
        logger.info('Transform')
        csv_filename, data_root, label_names = args.csv, args.data_root, args.label_names
        df = pd.read_csv(csv_filename)
        from img_utils import resize_height
        from point_utils import flatten_nest, partition
        import transforms
        plt.figure(figsize=(7, 7))
        i_subplot = 1
        for _, row in df[:args.rows].iterrows():
            x, kps = load_data(row, Path(data_root), label_names)
            x, kps = resize_height(args.height, x, kps)
            sizes = [len(ps) for ps in kps]
            flat_points = flatten_nest(kps)
            for _ in range(args.columns):
                tx, tkps = transforms.random_transform(x, flat_points)
                tx = tx.squeeze()
                tkps = partition(tkps, sizes)
                tkps = np.array(tkps)
                plt.subplot(args.rows, args.columns, i_subplot)
                i_subplot += 1
                plt.imshow(tx, 'gray')
                for pts in tkps:
                    plt.scatter(pts[:, 1], pts[:, 0], s=2)
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