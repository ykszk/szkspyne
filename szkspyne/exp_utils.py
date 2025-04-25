import json
import sys
from pathlib import Path
from typing import List, Sequence, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed, parallel_backend
from logzero import logger
from PIL import Image
from tqdm import tqdm
import pydicom

from szkspyne import img_utils, transforms
from szkspyne.settings import LABEL_NAMES


def load_dicom(filename: Union[str, Path]) -> np.ndarray:
    dcm = pydicom.dcmread(filename)
    x = dcm.pixel_array
    if dcm.PhotometricInterpretation == 'MONOCHROME1':
        x = np.amax(x) - x
    # x = (x - np.min(x)) / (np.max(x) - np.min(x))
    # convert to uint16
    x = x - np.min(x)
    x = np.clip(x, 0, 2**16 - 1)
    x = x.astype(np.uint16)

    return x


def plot_history(df):
    melted = pd.melt(df, value_vars=['train_loss', 'val_loss'], id_vars=['step'], value_name='loss', var_name='subset')
    sns.lineplot(data=melted, x='step', y='loss', hue='subset')


def load_data(row, data_root: Optional[Path], label_names: Sequence[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
    '''Load data from a row. Keypoints should be np.ndarray'''
    if data_root is None:
        img_path = row.imagePath
    else:
        img_path = data_root / row.imagePath
    if Path(img_path).suffix == '.dcm':
        x = load_dicom(img_path)
    else:
        x = np.array(Image.open(img_path))
    kps = [row.get(l) for l in label_names]
    # flip kps to change format from (x, y) to (y, x)
    kps = [np.flip(kp, axis=-1) if isinstance(kp, np.ndarray) else np.array(()) for kp in kps]
    return x, kps


def crop_by_kps(x, kps, crop_indices, margin):
    '''
    margin: 0 to 1
    '''
    cropping_kps = [kps[i] for i in crop_indices if len(kps[i]) > 0]
    tl = np.min(np.concatenate(cropping_kps, axis=0), axis=0)
    br = np.max(np.concatenate(cropping_kps, axis=0), axis=0)
    margin = (br - tl) * margin
    tl = tl - margin
    br = br + margin
    tl = np.clip(tl, 0, x.shape[:2]).astype(int)
    br = np.clip(br, 0, x.shape[:2]).astype(int)
    # logger.warning('Crop: %s %s %s %s', tl, br, x.shape, row.imagePath)
    cropped_x = x[tl[0] : br[0], tl[1] : br[1]]
    cropped_kp = [(kp - tl) if len(kp) > 0 else kp for kp in kps]
    return cropped_x, cropped_kp


def load_dataset(
    df: pd.DataFrame,
    height: int,
    data_root: Optional[Path],
    label_names: Sequence[str],
    num_workers: int,
    down_factor: int,
    clahe: bool,
    crop_label_names: Sequence[str] = [],
):
    '''
    load images and keypoints
    '''
    crop_indices = [label_names.index(c) for c in crop_label_names if c in label_names]
    logger.info(f'Crop labels and indices: {crop_label_names} {crop_indices}')

    def load(row):
        x, kp = load_data(row, data_root, label_names)
        assert kp[0].size != 0
        ret_x, ret_kp = [], []
        if crop_indices:
            margin = 0.1  # TODO: magic number
            cropped_x, cropped_kp = crop_by_kps(x, kp, crop_indices, margin)
            cropped_x, cropped_kp = img_utils.resize_height(height, cropped_x, cropped_kp)
            ret_x.append(cropped_x)
            ret_kp.append(cropped_kp)
        x, kp = img_utils.resize_height(height, x, kp)
        ret_x.append(x)
        ret_kp.append(kp)
        if clahe:
            ret_x = [transforms.apply_clahe(x) for x in ret_x]
        return ret_x, ret_kp

    # default backend gets frozen due to pytorch lightning?
    with parallel_backend('threading', n_jobs=num_workers):
        data = Parallel()(delayed(load)(row) for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Load data'))

    xs, kps = zip(*data)
    # flatten nested lists
    xs = [x for sublist in xs for x in sublist]
    kps = [kp for sublist in kps for kp in sublist]

    shapes = [x.shape for x in xs]
    max_shape = np.max(shapes, axis=0)

    new_shape = (down_factor * np.ceil(np.array(max_shape) / down_factor)).astype(np.int64).tolist()
    xs = [img_utils.pad_to_shape(x, new_shape, 0) for x in xs]
    return xs, kps


def ndjson2df(filename: Union[str, Path]) -> pd.DataFrame:
    with open(filename) as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        data_line = json.loads(line)
        data = data_line['content']
        shapes = data['shapes']
        points = {k: np.array(p)[:, 0] for k, p in shapes['point'].items() if k in LABEL_NAMES}
        for name in LABEL_NAMES:
            if name not in points.keys():
                points[name] = np.array(())
        row = {
            'filename': data_line['filename'],
            'imagePath': data['imagePath'],
            'flags': data['flags'],
            'group_id': data.get('group_id', None),
        }
        row.update(points)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def main():
    import argparse

    from logzero import logger

    parser = argparse.ArgumentParser(description='Try utility functions for debugging.')
    parser.add_argument('csv', help='CSV or NDJSON filename: %(default)s', nargs='?', default='../script/verse.csv')
    parser.add_argument('--data_root', help='Data root directory: %(default)s', nargs='?')
    parser.add_argument('--label_names', help='Label names: %(default)s', nargs='?', default=LABEL_NAMES)
    parser.add_argument('--height', help='Resized image height: %(default)d', default=1024)

    subparsers = parser.add_subparsers()
    p_summary = subparsers.add_parser('load_data', help="load a set of image and keypoints from a csv file.")
    p_summary.add_argument('--output', help='Optional output image. e.g. keypoints.svg')

    def command_load_data(args):
        csv_filename, data_root, label_names = args.csv, args.data_root, args.label_names
        if csv_filename.endswith('.ndjson'):
            df = ndjson2df(csv_filename)
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
                plt.gcf().savefig(args.output, transparent=True, bbox_inches='tight')
            else:
                plt.show()
            plt.close()
            break
        return 0

    p_summary.set_defaults(handler=command_load_data)

    p_transform = subparsers.add_parser('transform', help="Test transforms.random_transform.")
    p_transform.add_argument('--rows', help='The number of rample data: %(default)d', type=int, default=5)
    p_transform.add_argument(
        '--columns', help='The number of random transformations: %(default)d', type=int, default=5
    )
    p_transform.add_argument('--output', help='Optional output image. e.g. random_transforms.svg')

    def command_transform(args):
        logger.info('Transform')
        csv_filename, data_root, label_names = args.csv, args.data_root, args.label_names
        if csv_filename.endswith('.ndjson'):
            df = ndjson2df(csv_filename)
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
            plt.gcf().savefig(args.output, transparent=True, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        return 0

    p_transform.set_defaults(handler=command_transform)

    p_loader = subparsers.add_parser('loader', help="Test KeypointDataset.")
    p_loader.add_argument('--rows', help='The number of rample data: %(default)d', type=int, default=5)
    p_loader.add_argument('--sigma', help='Sigma for heatmaps: %(default)d', type=int, default=5)
    p_loader.add_argument(
        '--transform', help='Transformation: %(default)s', choices=['random', 'base'], default='random'
    )
    p_loader.add_argument('--output', help='Optional output image. e.g. kp_dataset.svg')
    p_loader.add_argument('--mhd_x', help='Optional output mhd image for training input. e.g. kp_dataset.mhd')
    p_loader.add_argument('--mhd_y', help='Optional output mhd image for training output. e.g. kp_dataset.mhd')
    p_loader.add_argument('-l', '--crop_label_names', help='Label names for cropping', nargs='*', type=str)
    p_loader.add_argument('--clahe', help='Apply CLAHE', action='store_true')

    def command_loader(args):
        logger.info('Loader')
        csv_filename, data_root, label_names = args.csv, args.data_root, args.label_names
        if csv_filename.endswith('.ndjson'):
            df = ndjson2df(csv_filename)
        else:
            df = pd.read_csv(csv_filename)
        import settings
        import transforms
        from img_utils import pad_to_shape, resize_height
        from point_utils import KeypointDataset, flatten_nest, partition
        from torch.utils.data import DataLoader

        logger.info("Crop label names: %s", args.crop_label_names)

        flip_pairs = settings.FLIP_PAIRS
        down_factor = 256
        crop_label_names = args.crop_label_names if args.crop_label_names else []
        xs, kps = load_dataset(
            df.sample(n=args.rows),
            args.height,
            data_root,
            label_names,
            4,
            down_factor,
            args.clahe,
            crop_label_names,
        )

        print(len(xs), len(kps))

        tf = transforms.random_transform if args.transform == 'random' else transforms.base_kps_transform
        dataset = KeypointDataset(xs, kps, None, flip_pairs, tf, sigma=args.sigma)
        loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=len(xs),
            num_workers=4,
        )
        plt.figure(figsize=(10, 10))
        for batch_xs, batch_ys in loader:
            # for i_subplot, (x, y) in enumerate(zip(batch_xs, batch_ys)):
            #     plt.subplot(args.rows, len(y) + 1, i_subplot * (len(y) + 1) + 1)
            #     plt.imshow(x.squeeze(), 'gray')
            #     plt.axis('off')
            #     for i, hm in enumerate(y):
            #         u8img = np.round(np.clip(hm.numpy(), 0, 1) * 255).astype(np.uint8)
            #         plt.subplot(args.rows, len(y) + 1, i_subplot * (len(y) + 1) + i + 2)
            #         plt.imshow(u8img, 'gray')
            #         plt.axis('off')
            if args.mhd_x:
                from szkmipy import mhd

                logger.info('Save x batch')
                print('batch shape in utils', batch_xs.shape)
                batch_xs = batch_xs.squeeze()  # remove channel dim
                mhd.write(args.mhd_x, batch_xs.cpu().numpy())
            if args.mhd_y:
                from szkmipy import mhd

                logger.info('Save y batch')
                # batch_ys = batch_ys.reshape((-1, batch_ys.shape[2], batch_ys.shape[3]))  # merge channel and batch dims
                # to channel last
                batch_ys = batch_ys.permute(0, 2, 3, 1)
                h = {'ElementNumberOfChannels': batch_ys.shape[-1]}
                mhd.write(args.mhd_y, batch_ys.cpu().numpy(), h)
        if args.output:
            plt.gcf().savefig(args.output, transparent=True, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        return 0

    p_loader.set_defaults(handler=command_loader)

    p_test_loader = subparsers.add_parser('test_loader', help="Test loader by loading all data with augmentation.")
    p_test_loader.add_argument('--sigma', help='Sigma for heatmaps: %(default)d', type=int, default=5)
    p_test_loader.add_argument('--batch_size', help='Batch size: %(default)d', type=int, default=8)
    p_test_loader.add_argument('--workers', help='Number of workers: %(default)d', type=int, default=8)
    p_test_loader.add_argument(
        '--transform', help='Transformation: %(default)s', choices=['random', 'base'], default='random'
    )
    p_test_loader.add_argument('-l', '--crop_label_names', help='Label names for cropping', nargs='*', type=str)
    p_test_loader.add_argument('--clahe', help='Apply CLAHE', action='store_true')

    def command_test_loader(args):
        logger.info('Test Loader')
        csv_filename, data_root, label_names = args.csv, args.data_root, args.label_names
        if csv_filename.endswith('.ndjson'):
            df = ndjson2df(csv_filename)
        else:
            df = pd.read_csv(csv_filename)
        import settings
        import transforms
        from point_utils import KeypointDataset
        from torch.utils.data import DataLoader

        logger.info("Crop label names: %s", args.crop_label_names)

        flip_pairs = settings.FLIP_PAIRS
        down_factor = 256
        crop_label_names = args.crop_label_names if args.crop_label_names else []
        logger.info("Load dataset")
        print(df.iloc[238].filename)
        xs, kps = load_dataset(
            df,
            args.height,
            data_root,
            label_names,
            args.workers,
            down_factor,
            args.clahe,
            crop_label_names,
        )

        print(len(xs), len(kps))

        tf = transforms.random_transform if args.transform == 'random' else transforms.base_kps_transform
        dataset = KeypointDataset(xs, kps, None, flip_pairs, tf, sigma=args.sigma)
        loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        logger.info("Test loader")
        for i, (batch_xs, batch_ys) in enumerate(loader):
            print(i, batch_xs.shape, batch_ys.shape)
        return 0

    p_test_loader.set_defaults(handler=command_test_loader)

    args = parser.parse_args()
    if not hasattr(args, 'handler'):
        print('No command was specified.')
        parser.print_help()
        sys.exit(1)
    return args.handler(args)


if __name__ == '__main__':
    sys.exit(main())
