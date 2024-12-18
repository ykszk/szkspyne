import argparse
import sys
from pathlib import Path
from typing import List
import itertools
import scipy.stats
import numpy as np
import logzero
from logzero import logger
from typing import Union

from szkmipy import mhd
from szkspyne import settings
from szkspyne.heatmap2labelme import save_kps_json


PROB_MAX = settings.PROB_MAX


def worker(task_args):
    (pred, original_shape, image_filename, json_filename, hm_filename, ch_as_z, clahe) = task_args
    if clahe:
        # flatten x and y
        flattened = pred.reshape(pred.shape[0], pred.shape[1], -1)
        entropies = scipy.stats.entropy(flattened, base=2, axis=2)
        min_idx = np.argmin(entropies, axis=0)
        pred = pred[min_idx, np.arange(pred.shape[1])]
    pred *= PROB_MAX
    pred = pred.astype(np.uint16)
    if json_filename:
        save_kps_json(
            pred,
            original_shape,
            image_filename,
            json_filename,
            settings.LABEL_NAMES,
            min_thresh=0.1 * PROB_MAX,
        )
    if hm_filename:
        if ch_as_z:
            mhd.write(hm_filename, pred)
        else:
            pred = np.transpose(pred, [1, 2, 0])
            mhd.write(hm_filename, pred, {'ElementNumberOfChannels': pred.shape[-1]})


def main():
    parser = argparse.ArgumentParser(description='Apply spine model.')

    subparsers = parser.add_subparsers()

    def add_model_params(parser):
        model_group = parser.add_mutually_exclusive_group(required=True)
        model_group.add_argument('--ckpt', help='Model checkpoint')
        model_group.add_argument('--hparams', help='Model hyperparameters in pickle')
        parser.add_argument('--weights', help='Model weights. Required for --hparams option')
        return model_group

    p_single = subparsers.add_parser('single', help="Apply to a single image.")
    p_single.add_argument('input', help='Input mhd filename:', metavar='<input>')
    add_model_params(p_single)
    p_single.add_argument('--preproc', help='Save preprocessed input image')
    p_single.add_argument('--labelme', help='Save labelme json file')
    p_single.add_argument('--heatmap', help='Save heatmap file')
    p_single.add_argument('--clahe', help='Clahe in preprocessing', action='store_true')
    p_single.add_argument('--height', help='Image height: %(default)s', metavar='<int>', type=int, default=768)
    p_single.add_argument('--ch_as_z', help='Use channel as z axis', action='store_true')
    p_single.set_defaults(mode='single')  # not using func to enable lazy loading of the modules

    p_batch = subparsers.add_parser('batch', help="Apply to a batch of images.")
    p_batch.add_argument('--output', help='Output directory', type=Path)

    p_batch.add_argument('input', help='Input directory or text file containing files', metavar='<input>')
    add_model_params(p_batch)
    p_batch.add_argument('--ext', help='Input image file extension: %(default)s', default='.png')
    p_batch.add_argument('--labelme', help='Save labelme json file', action='store_true')
    p_batch.add_argument('--heatmap', help='Save heatmap', action='store_true')
    p_batch.add_argument('--keep_image_path', help='Keep image path', action='store_true')
    p_batch.add_argument('--clahe', help='Clahe in preprocessing', action='store_true')
    p_batch.add_argument('--height', help='Image height: %(default)s', metavar='<int>', type=int, default=768)
    p_batch.add_argument('--workers', help='The number of workers: %(default)s', metavar='<int>', type=int, default=4)
    p_batch.add_argument('--ch_as_z', help='Use channel as z axis', action='store_true')
    p_batch.add_argument('--batch_size', help='Batch size: %(default)s', metavar='<int>', type=int, default=4)
    p_batch.set_defaults(mode='batch')

    args = parser.parse_args()

    log_format = '%(color)s[%(levelname)1.1s %(asctime)s.%(msecs)d %(module)s:%(lineno)d]%(end_color)s %(message)s'
    formatter = logzero.LogFormatter(fmt=log_format)
    logzero.setup_default_logger(formatter=formatter)
    logger.info('Load modules')

    # lazy import
    import cv2
    import img_utils
    import pretrained_models
    import pydicom
    import torch
    import transforms
    from PIL import Image
    from functools import lru_cache

    DOWN_FACTOR = 256

    # load with cache
    @lru_cache(maxsize=4)
    def _load(filename: Union[str, Path]):
        if str(filename).endswith('.dcm') or str(filename).endswith('.dicom'):
            dcm = pydicom.dcmread(filename)
            x = dcm.pixel_array
            if dcm.PhotometricInterpretation == 'MONOCHROME1':
                x = x.max() - x
        else:
            img = Image.open(filename)
            x = np.array(img)
            if img.mode == 'I':
                x = x.astype(np.uint16)
        if x.ndim == 3:
            x = x[..., 0]
        return x

    def load(filename: Union[str, Path], height, pad: bool, clahe: bool):
        x = _load(filename)
        original_shape = x.shape
        # assert x.dtype == np.uint16, 'np.uint16 is expected'
        x = img_utils.resize_height(height, x)
        assert isinstance(x, np.ndarray)
        if pad:
            x = img_utils.pad_to_next_multiple(DOWN_FACTOR, x, [1], 0)
        # logger.info('Padded to %s', x.shape)
        if clahe:
            x = apply_clahe(x)
        x = transforms.percentile_normalize(x, 1)  # , clip=True)
        x = x.astype(np.float32)
        return x, original_shape

    def apply_clahe(img, clipLimit=100, tileGridSize=(16, 16)):
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)  # type: ignore
        cl = clahe.apply(img)
        return cl

    def load_model(args, map_location=None) -> pretrained_models.LitSMP:
        if args.hparams and not args.weights:
            print('--weights is required for --hparams')
            sys.exit(1)
        if args.ckpt:
            lit_model = pretrained_models.LitSMP.load_from_checkpoint(args.ckpt, map_location=map_location)
            assert isinstance(lit_model, pretrained_models.LitSMP)
        else:
            import pickle

            with open(args.hparams, 'rb') as f:
                hparams = pickle.load(f)
            lit_model = pretrained_models.LitSMP(hparams.in_chs, hparams.out_chs, hparams.hparams)
            lit_model.model.load_state_dict(torch.load(args.weights, map_location=map_location))
        return lit_model

    def apply_single(args):
        # device = 'cpu'
        torch.set_num_threads(8)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        logger.info('Use %s', device)
        logger.info('load model')
        lit_model = load_model(args, device)
        lit_model.eval()
        logger.info('load image')
        x, original_shape = load(args.input, args.height, True, args.clahe)
        if args.preproc:
            mhd.write(args.preproc, x)
        assert isinstance(x, np.ndarray)
        if x.ndim == 2:
            x = x[np.newaxis]
        # x = torch.from_numpy(x[np.newaxis], )
        # x = torch.tensor(x, pin_memory=True)
        logger.info('Apply model')
        x = torch.as_tensor(x[np.newaxis], device=device)
        # x = x.to(device)
        lit_model.to(device)
        with torch.no_grad():
            pred = torch.sigmoid(lit_model(x))
            # pred = lit_model(x)
        logger.info('Done')
        pred = pred.to('cpu').detach().numpy().copy()[0]
        logger.info('Pred min: %s, Pred max: %s', pred.min(), pred.max())
        if args.labelme:
            logger.info('Save %s', args.labelme)
            save_kps_json(
                pred,
                original_shape,
                Path(args.input).name,
                args.labelme,
                settings.LABEL_NAMES,
                min_thresh=0.1 * PROB_MAX,
            )
        if args.heatmap:
            if args.ch_as_z:
                mhd.write(args.heatmap, pred)
            else:
                pred = np.transpose(pred, [1, 2, 0])
                mhd.write(args.heatmap, pred, {'ElementNumberOfChannels': pred.shape[-1]})

    def pad_batch(xs, factor, value):
        shapes = [x.shape for x in xs]
        max_shape = np.max(shapes, axis=0)
        new_shape = (factor * np.ceil(np.array(max_shape) / factor)).astype(np.int64).tolist()
        xs = [img_utils.pad_to_shape(x, new_shape, value)[np.newaxis] for x in xs]
        return xs

    def apply_batch(args):
        import tqdm
        from joblib import Parallel, delayed

        logger.info('Enumerate input files')
        indir = Path(args.input)
        if indir.is_file():
            with open(indir) as f:
                input_filenames = [Path(line.strip()) for line in f]
        else:
            input_filenames = sorted(indir.glob('*' + args.ext))
        if len(input_filenames) == 0:
            print("No image file found in ", args.input)
            return 1
        from torch.utils.data import DataLoader, Dataset

        class NDArray(Dataset):
            def __init__(self, filenames: List[Path]):
                self.filenames = filenames

            def __getitem__(self, index):
                filename = self.filenames[index]
                arr, original_shape = load(filename, args.height, True, args.clahe)
                if args.clahe:
                    original_arr, _ = load(filename, args.height, True, False)
                    arr = [arr, original_arr]
                return arr, (filename, original_shape)

            def __len__(self):
                return len(self.filenames)

        def collate_fn(batch):
            xs, aux = zip(*batch)
            if args.clahe:
                # flatten
                xs = list(itertools.chain.from_iterable(xs))
                # logger.debug('Batch size: %s', len(xs))
            xs = pad_batch(xs, DOWN_FACTOR, 0)
            # logger.debug("Aux: %s", aux)
            return torch.from_numpy(np.stack(xs, axis=0)), zip(*aux)

        loader = DataLoader(
            NDArray(input_filenames),
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate_fn,
            shuffle=False,
        )
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        logger.info('load model')
        lit_model = load_model(args, device)
        lit_model.to(device)
        lit_model.eval()

        task_pbar = tqdm.tqdm(total=len(input_filenames))
        logger.info('Apply model')

        def callback(_):
            task_pbar.update()

        from multiprocessing import Pool

        def shannon_entropy(p: torch.Tensor, dim: int) -> torch.Tensor:
            p = p / torch.sum(p, dim=dim, keepdim=True)
            p = torch.clip(p, 1.0e-8, None)
            logp = torch.log2(p)
            e = -torch.sum(p * logp, dim=dim)
            return e

        with Pool(processes=args.workers) as pool:
            for batch, (filenames, original_shapes) in tqdm.tqdm(loader):
                batch = batch.to(device)
                with torch.no_grad():
                    pred = torch.sigmoid(lit_model(batch))
                    if args.clahe:
                        # from (batch and raw_or_clahe, channel, height, width) to (batch, raw_or_clahe, channel, height, width)
                        # logger.debug('Pred shape: %s', pred.shape)
                        pred = pred.reshape(pred.shape[0] // 2, 2, *pred.shape[1:])
                        # logger.debug('Reshaped pred shape: %s', pred.shape)
                    #     # calculate entropy per channel
                    #     shape_flat_xy = list(pred.shape[:-2]) + [-1]
                    #     pred_flat_xy = pred.reshape(shape_flat_xy)
                    #     entropy_per_ch = shannon_entropy(pred_flat_xy, dim=-1)
                    #     min_idx = torch.argmin(entropy_per_ch, dim=1)
                    #     # select the channel with the lowest entropy
                    #     pred = pred[torch.arange(pred.shape[0]), min_idx]

                    # pred *= PROB_MAX
                    pred = pred.to('cpu').detach().numpy().astype(np.float32)

                filenames = [Path(fn) for fn in filenames]
                # original_shapes = list(zip(*[t.to('cpu').tolist() for t in original_shapes]))
                if args.keep_image_path:
                    image_filenames = [str(fn) for fn in filenames]
                else:
                    image_filenames = [fn.name for fn in filenames]
                json_filenames = [fn.with_suffix('.json') if args.labelme else None for fn in filenames]
                hm_filenames = [fn.with_suffix('.mha') if args.heatmap else None for fn in filenames]
                if args.output:
                    json_filenames = [args.output / fn.with_suffix('.json').name for fn in filenames]
                    hm_filenames = [args.output / fn.with_suffix('.mha').name for fn in filenames]
                for task_args in zip(pred, original_shapes, image_filenames, json_filenames, hm_filenames):
                    task_args = list(task_args) + [args.ch_as_z, args.clahe]
                    pool.apply_async(worker, (task_args,), callback=callback, error_callback=print)
            pool.close()
            pool.join()
        logger.info('Done applying')

    if args.mode == 'single':
        logger.info('Apply single')
        apply_single(args)
    elif args.mode == 'batch':
        logger.info('Apply batch')
        apply_batch(args)
    else:
        raise RuntimeError('Invalid command')


if __name__ == '__main__':
    sys.exit(main())
