import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Tuple
from pathlib import Path
from PIL import Image


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

    import argparse
    parser = argparse.ArgumentParser(
        description='Try utility functions for debugging.')
    subparsers = parser.add_subparsers()
    p_summary = subparsers.add_parser(
        'load_data', help="load a set of image and keypoints from a csv file.")
    p_summary.add_argument('csv',
                           help='CSV filename: %(default)s',
                           nargs='?',
                           default='../script/verse.csv')
    p_summary.add_argument('data_root',
                           help='Data root directory: %(default)s',
                           nargs='?',
                           default='../../data')
    p_summary.add_argument('label_names',
                           help='Label names: %(default)s',
                           nargs='?',
                           default=['TL', 'TR', 'BL', 'BR'])
    p_summary.add_argument('--output',
                           help='Optional output image. e.g. kps.svg')

    def command_load_data(args):
        csv_filename, data_root, label_names = args.csv, args.data_root, args.label_names
        df = pd.read_csv(csv_filename)
        for _, row in df.iterrows():
            x, kps = load_data(row, Path(data_root), label_names)
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
    # parser.add_argument('-o',
    #                     '--option',
    #                     help='Optional argument. default: %(default)s',
    #                     metavar='<name>',
    #                     default='default')
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