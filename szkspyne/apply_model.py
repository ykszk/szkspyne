import sys
import argparse

PROB_MAX = 4000


def main():
    parser = argparse.ArgumentParser(description='Apply spine model.')

    subparsers = parser.add_subparsers()
    p_single = subparsers.add_parser('single', help="Apply to a single image.")
    p_single.add_argument('input',
                          help='Input mhd filename:',
                          metavar='<input>')
    p_single.add_argument('model', help='Model checkpoint', metavar='<model>')
    # p_single.add_argument('output',
    #                       help='Output mhd filename:',
    #                       metavar='<output>')
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

    p_batch.add_argument('input',
                         help='Input mhd filename:',
                         metavar='<input>')
    p_batch.add_argument('model', help='Model checkpoint', metavar='<model>')
    # p_batch.add_argument('output',
    #                      help='Output mhd filename:',
    #                      metavar='<output>')
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
    args = parser.parse_args()
    # lazy import
    from pathlib import Path
    import numpy as np
    from logzero import logger
    import pretrained_models
    import transforms
    import mhd
    import point_utils
    import img_utils
    import labelme_utils
    import settings

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
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        cl = clahe.apply(img)
        return cl

    def save_kps_json(pred,
                      original_shape,
                      image_filename,
                      json_filename,
                      min_thresh=.25):
        kps_per_ch = point_utils.local_center_of_masses(pred, min_thresh)
        # half_shape = (np.array(original_shape, dtype=np.float32) // 2).tolist()
        # scale = half_shape[0] / pred.shape[1]
        scale = original_shape[0] / pred.shape[1]
        kps_per_ch = [(np.array(kps, dtype=np.float32) * scale).tolist()
                      for kps in kps_per_ch]
        json_data = labelme_utils.kps2labelme(kps_per_ch, image_filename,
                                              original_shape,
                                              settings.LABEL_NAMES)
        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=2)

    def apply_single(args):
        logger.info('load model')
        lit_model = pretrained_models.LitSMP.load_from_checkpoint(args.model)
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
                          Path(args.input).name, args.labelme)
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
        lit_model = pretrained_models.LitSMP.load_from_checkpoint(args.model)
        lit_model.to(device)
        lit_model.eval()

        logger.info('Apply model')
        for batch, (filenames, original_shapes) in tqdm.tqdm(loader):
            batch = batch.to(device)
            with torch.no_grad():
                pred = torch.sigmoid(lit_model(batch))
                pred *= PROB_MAX
                pred = pred.to('cpu').detach().numpy().astype(np.uint16)

            filenames = [Path(fn) for fn in filenames]
            original_shapes = list(
                zip(*[t.to('cpu').tolist() for t in original_shapes]))
            if args.labelme:
                json_filenames = [fn.with_suffix('.json') for fn in filenames]
                image_filenames = [fn.name for fn in filenames]

                def save_json(args):
                    return save_kps_json(*args, min_thresh=.1 * PROB_MAX)

                Parallel(n_jobs=args.workers)(
                    delayed(save_json)(arg)
                    for arg in zip(pred, original_shapes, image_filenames,
                                   json_filenames))

            if args.heatmap:

                def save_hm(pred, fn):
                    pred = np.transpose(pred, [1, 2, 0])
                    mhd.write(fn, pred,
                              {'ElementNumberOfChannels': pred.shape[-1]})

                hm_filenames = [fn.with_suffix('.mha') for fn in filenames]
                Parallel(n_jobs=args.workers)(
                    delayed(save_hm)(*arg) for arg in zip(pred, hm_filenames))

    if args.mode == 'single':
        apply_single(args)
    elif args.mode == 'batch':
        apply_batch(args)
    else:
        raise RuntimeError('Invalid command')


if __name__ == '__main__':
    sys.exit(main())
