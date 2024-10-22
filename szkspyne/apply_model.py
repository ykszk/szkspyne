import argparse
import sys

import numpy as np
from szkmipy import mhd

from szkspyne import settings
from szkspyne.heatmap2labelme import save_kps_json


PROB_MAX = settings.PROB_MAX


def worker(task_args):
    (pred, original_shape, image_filename, json_filename, hm_filename, ch_as_z) = task_args
    if json_filename:
        save_kps_json(
            pred, original_shape, image_filename, json_filename, settings.LABEL_NAMES, min_thresh=0.1 * PROB_MAX
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

    p_batch = subparsers.add_parser('batch', help="Apply to a single image.")
    p_batch.add_argument('--output', help='Optional output image. e.g. keypoints.svg')

    p_batch.add_argument('input', help='Input directory:', metavar='<input>')
    add_model_params(p_batch)
    p_batch.add_argument('--ext', help='Input image file extension: %(default)s', default='.png')
    p_batch.add_argument('--labelme', help='Save labelme json file', action='store_true')
    p_batch.add_argument('--heatmap', help='Save heatmap', action='store_true')
    p_batch.add_argument('--clahe', help='Clahe in preprocessing', action='store_true')
    p_batch.add_argument('--height', help='Image height: %(default)s', metavar='<int>', type=int, default=768)
    p_batch.add_argument('--workers', help='The number of workers: %(default)s', metavar='<int>', type=int, default=4)
    p_batch.add_argument('--ch_as_z', help='Use channel as z axis', action='store_true')
    p_batch.add_argument('--batch_size', help='Batch size: %(default)s', metavar='<int>', type=int, default=4)
    p_batch.set_defaults(mode='batch')

    args = parser.parse_args()

    # lazy import
    import logzero
    from logzero import logger

    log_format = '%(color)s[%(levelname)1.1s %(asctime)s.%(msecs)d %(module)s:%(lineno)d]%(end_color)s %(message)s'
    formatter = logzero.LogFormatter(fmt=log_format)
    logzero.setup_default_logger(formatter=formatter)
    logger.info('Load modules')

    from pathlib import Path
    from typing import Union

    import cv2
    import img_utils
    import pretrained_models
    import pydicom
    import torch
    import transforms
    from PIL import Image

    DOWN_FACTOR = 256

    def load(filename: Union[str, Path], height, pad: bool, clahe: bool):
        # x, _ = mhd.read(filename)
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
        original_shape = x.shape
        # assert x.dtype == np.uint16, 'np.uint16 is expected'
        x = img_utils.resize_height(height, x)
        assert isinstance(x, np.ndarray)
        if pad:
            x = img_utils.pad_to_next_multiple(DOWN_FACTOR, x, [1], 0)
        logger.info('Padded to %s', x.shape)
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

    def apply_batch(args):
        import tqdm
        from joblib import Parallel, delayed

        logger.info('load images')
        indir = Path(args.input)
        n_jobs = args.workers
        input_filenames = sorted(indir.glob('*' + args.ext))
        if len(input_filenames) == 0:
            print("No image file found in ", args.input)
            return 1
        xs, original_shapes = zip(
            *Parallel(n_jobs=n_jobs)(
                delayed(load)(fn, args.height, False, args.clahe) for fn in tqdm.tqdm(input_filenames)
            )
        )
        shapes = [x.shape for x in xs]
        max_shape = np.max(shapes, axis=0)
        new_shape = (DOWN_FACTOR * np.ceil(np.array(max_shape) / DOWN_FACTOR)).astype(np.int64).tolist()
        logger.info('Pad to %s', new_shape)
        xs = [img_utils.pad_to_shape(x, new_shape, 0)[np.newaxis] for x in xs]
        from torch.utils.data import DataLoader, Dataset

        class NDArray(Dataset):
            def __init__(self, arr, aux):
                assert len(arr) == len(aux)
                self.arr = arr
                self.aux = aux

            def __getitem__(self, index):
                return self.arr[index], self.aux[index]

            def __len__(self):
                return len(self.arr)

        loader = DataLoader(
            NDArray(xs, [(str(fn), o_s) for fn, o_s in zip(input_filenames, original_shapes)]),
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,
        )
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        logger.info('load model')
        lit_model = load_model(args, device)
        lit_model.to(device)
        lit_model.eval()

        task_pbar = tqdm.tqdm(total=len(xs))
        logger.info('Apply model')

        def callback(_):
            task_pbar.update()

        from multiprocessing import Pool

        with Pool(processes=args.workers) as pool:
            for batch, (filenames, original_shapes) in tqdm.tqdm(loader):
                batch = batch.to(device)
                with torch.no_grad():
                    pred = torch.sigmoid(lit_model(batch))
                    pred *= PROB_MAX
                    pred = pred.to('cpu').detach().numpy().astype(np.uint16)

                filenames = [Path(fn) for fn in filenames]
                original_shapes = list(zip(*[t.to('cpu').tolist() for t in original_shapes]))
                image_filenames = [fn.name for fn in filenames]
                json_filenames = [fn.with_suffix('.json') if args.labelme else None for fn in filenames]
                hm_filenames = [fn.with_suffix('.mha') if args.heatmap else None for fn in filenames]
                for task_args in zip(pred, original_shapes, image_filenames, json_filenames, hm_filenames):
                    task_args = list(task_args) + [args.ch_as_z]
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
