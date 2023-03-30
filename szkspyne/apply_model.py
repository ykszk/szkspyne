import sys
import argparse

PROB_MAX = 4000

# Cannot lazy load for pickling
import json
import numpy as np
import point_utils
import labelme_utils
import mhd
import settings


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


def worker(task_args):
    (pred, original_shape, image_filename, json_filename,
     hm_filename) = task_args
    if json_filename:
        save_kps_json(pred,
                      original_shape,
                      image_filename,
                      json_filename,
                      settings.LABEL_NAMES,
                      min_thresh=.1 * PROB_MAX)
    if hm_filename:
        pred = np.transpose(pred, [1, 2, 0])
        mhd.write(hm_filename, pred,
                  {'ElementNumberOfChannels': pred.shape[-1]})


def main():
    parser = argparse.ArgumentParser(description='Apply spine model.')

    subparsers = parser.add_subparsers()

    def add_model_params(parser):
        model_group = parser.add_mutually_exclusive_group(required=True)
        model_group.add_argument('--ckpt', help='Model checkpoint')
        model_group.add_argument('--hparams',
                                 help='Model hyperparameters in pickle')
        parser.add_argument(
            '--weights', help='Model weights. Required for --hparams option')
        return model_group

    p_single = subparsers.add_parser('single', help="Apply to a single image.")
    p_single.add_argument('input',
                          help='Input mhd filename:',
                          metavar='<input>')
    add_model_params(p_single)
    p_single.add_argument('--preproc', help='Save preprocessed input image')
    p_single.add_argument('--labelme', help='Save labelme json file')
    p_single.add_argument('--heatmap', help='Save heatmap file')
    p_single.add_argument('--height',
                          help='Image height: %(default)s',
                          metavar='<int>',
                          type=int,
                          default=768)
    p_single.set_defaults(
        mode='single')  # not using func to enable lazy loading of the modules

    p_batch = subparsers.add_parser('batch', help="Apply to a single image.")
    p_batch.add_argument('--output',
                         help='Optional output image. e.g. keypoints.svg')

    p_batch.add_argument('input', help='Input directory:', metavar='<input>')
    add_model_params(p_batch)
    p_batch.add_argument('--ext',
                         help='Input image file extension: %(default)s',
                         default='.png')
    p_batch.add_argument('--labelme',
                         help='Save labelme json file',
                         action='store_true')
    p_batch.add_argument('--heatmap', help='Save heatmap', action='store_true')
    p_batch.add_argument('--height',
                         help='Image height: %(default)s',
                         metavar='<int>',
                         type=int,
                         default=768)
    p_batch.add_argument('--workers',
                         help='The number of workers: %(default)s',
                         metavar='<int>',
                         type=int,
                         default=4)
    p_batch.add_argument('--batch_size',
                         help='Batch size: %(default)s',
                         metavar='<int>',
                         type=int,
                         default=4)
    p_batch.set_defaults(mode='batch')

    p_hm2lm = subparsers.add_parser(
        'hm2lm',
        help="Create labelme files from the given heatmaps (exp output).")
    p_hm2lm.add_argument('heatmap', help='Input heatmap filename')
    p_hm2lm.add_argument('ndjson', help='Input ndjson filename')
    p_hm2lm.add_argument('output', help='Output directory')
    p_hm2lm.add_argument('--save_heatmap',
                         help='Save heatmap next to json',
                         action='store_true')
    p_hm2lm.add_argument('--save_ndjson',
                         help='Save labelme data as an ndjson file')
    p_hm2lm.add_argument('--workers',
                         help='The number of workers: %(default)s',
                         metavar='<int>',
                         type=int,
                         default=4)
    p_hm2lm.set_defaults(mode='hm2lm')

    args = parser.parse_args()
    # lazy import
    from pathlib import Path
    from logzero import logger
    import pretrained_models
    import transforms
    import img_utils
    from typing import Union

    import torch
    import cv2
    from PIL import Image
    import json
    DOWN_FACTOR = 256

    def load(filename, height, pad: bool):
        # x, _ = mhd.read(filename)
        img = Image.open(filename)
        x = np.array(img)
        original_shape = x.shape
        if img.mode == 'I':
            x = x.astype(np.uint16)
        # assert x.dtype == np.uint16, 'np.uint16 is expected'
        x = img_utils.resize_height(height, x)
        assert isinstance(x, np.ndarray)
        if pad:
            x = img_utils.pad_to_next_multiple(DOWN_FACTOR, x, [1], 0)
        logger.info('Padded to %s', x.shape)
        x = apply_clahe(x)
        x = transforms.percentile_normalize(x, 1)  # , clip=True)
        x = x.astype(np.float32)
        return x, original_shape

    def apply_clahe(img, clipLimit=5000, tileGridSize=(32, 32)):
        clahe = cv2.createCLAHE(  # type: ignore
            clipLimit=clipLimit, tileGridSize=tileGridSize)
        cl = clahe.apply(img)
        return cl

    def load_model(args) -> pretrained_models.LitSMP:
        if args.hparams and not args.weights:
            print('--weights is required for --hparams')
            sys.exit(1)
        if args.ckpt:
            lit_model = pretrained_models.LitSMP.load_from_checkpoint(
                args.ckpt)
            assert isinstance(lit_model, pretrained_models.LitSMP)
        else:
            import pickle
            with open(args.hparams, 'rb') as f:
                hparams = pickle.load(f)
            lit_model = pretrained_models.LitSMP(hparams.in_chs,
                                                 hparams.out_chs,
                                                 hparams.hparams)
            lit_model.model.load_state_dict(torch.load(args.weights))
        return lit_model

    def apply_single(args):
        logger.info('load model')
        lit_model = load_model(args)
        lit_model.eval()
        logger.info('load image')
        x, original_shape = load(args.input, args.height, True)
        if args.preproc:
            mhd.write(args.preproc, x)
        assert isinstance(x, np.ndarray)
        if x.ndim == 2:
            x = x[np.newaxis]
        x = torch.from_numpy(x[np.newaxis])
        with torch.no_grad():
            pred = torch.sigmoid(lit_model(x))
            # pred = lit_model(x)
        pred = pred.to('cpu').detach().numpy().copy()[0]
        logger.info('Pred min: %s, Pred max: %s', pred.min(), pred.max())
        if args.labelme:
            logger.info('Save %s', args.labelme)
            save_kps_json(pred, original_shape,
                          Path(args.input).name, args.labelme,
                          settings.LABEL_NAMES)
        if args.heatmap:
            pred = np.transpose(pred, [1, 2, 0])
            mhd.write(args.heatmap, pred,
                      {'ElementNumberOfChannels': pred.shape[-1]})

    def apply_batch(args):
        from joblib import Parallel, delayed
        import tqdm
        logger.info('load images')
        indir = Path(args.input)
        n_jobs = args.workers
        input_filenames = sorted(indir.glob('*' + args.ext))
        xs, original_shapes = zip(*Parallel(n_jobs=n_jobs)(
            delayed(load)(fn, args.height, False)
            for fn in tqdm.tqdm(input_filenames)))
        shapes = [x.shape for x in xs]
        max_shape = np.max(shapes, axis=0)
        new_shape = (DOWN_FACTOR *
                     np.ceil(np.array(max_shape) / DOWN_FACTOR)).astype(
                         np.int64).tolist()
        logger.info('Pad to %s', new_shape)
        xs = [img_utils.pad_to_shape(x, new_shape, 0)[np.newaxis] for x in xs]
        from torch.utils.data import Dataset, DataLoader

        class NDArray(Dataset):

            def __init__(self, arr, aux):
                assert len(arr) == len(aux)
                self.arr = arr
                self.aux = aux

            def __getitem__(self, index):
                return self.arr[index], self.aux[index]

            def __len__(self):
                return len(self.arr)

        loader = DataLoader(NDArray(
            xs, [(str(fn), o_s)
                 for fn, o_s in zip(input_filenames, original_shapes)]),
                            batch_size=args.batch_size,
                            num_workers=args.workers,
                            shuffle=False)
        device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        logger.info('load model')
        lit_model = load_model(args)
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
                original_shapes = list(
                    zip(*[t.to('cpu').tolist() for t in original_shapes]))
                image_filenames = [fn.name for fn in filenames]
                json_filenames = [
                    fn.with_suffix('.json') if args.labelme else None
                    for fn in filenames
                ]
                hm_filenames = [
                    fn.with_suffix('.mha') if args.heatmap else None
                    for fn in filenames
                ]
                for task_args in zip(pred, original_shapes, image_filenames,
                                     json_filenames, hm_filenames):
                    pool.apply_async(worker, (task_args, ),
                                     callback=callback,
                                     error_callback=print)
            pool.close()
            pool.join()
        logger.info('Done applying')

    def heatmap2labelme(heatmap_filename: str, ndjson_filename: str,
                        outdir: Union[str, Path], save_heatmap: bool,
                        save_ndjson: Union[str, None], n_jobs: int):
        from joblib import Parallel, delayed
        import mhd
        outdir = Path(outdir)
        heatmaps, _ = mhd.read(heatmap_filename)
        heatmaps = heatmaps.transpose(
            (0, 3, 1, 2))  # from (B,Y,X,Ch) to (B,Ch,Y,X)
        with open(ndjson_filename) as f:
            json_data = [json.loads(l) for l in f.readlines()]

        def save_json(args):
            heatmap, data = args
            original_shape = data['imageHeight'], data['imageWidth']
            filepath = Path(data['filepath'])
            if filepath.parts[0] in ['CLO', 'SSS']:
                stem = filepath.stem
            else:
                stem = '_'.join(filepath.with_suffix('').parts[1:])
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

        data = Parallel(n_jobs=n_jobs)(delayed(save_json)(arg)
                                       for arg in zip(heatmaps, json_data))
        assert data is not None
        if save_ndjson:
            with open(save_ndjson, 'w') as f:
                for d in data:
                    f.write(json.dumps(d, ensure_ascii=False))
                    f.write('\n')

    if args.mode == 'single':
        apply_single(args)
    elif args.mode == 'batch':
        apply_batch(args)
    elif args.mode == 'hm2lm':
        heatmap2labelme(args.heatmap, args.ndjson, args.output,
                        args.save_heatmap, args.save_ndjson, args.workers)
    else:
        raise RuntimeError('Invalid command')


if __name__ == '__main__':
    sys.exit(main())
