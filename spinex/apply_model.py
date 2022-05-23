import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Apply lateral cervical spine model.')
    parser.add_argument('input', help='Input mhd filename:', metavar='<input>')
    parser.add_argument('model', help='Model checkpoint', metavar='<model>')
    parser.add_argument('output',
                        help='Output mhd filename:',
                        metavar='<output>')
    parser.add_argument('--preproc', help='Save preprocessed input image')
    parser.add_argument('--labelme', help='Save labelme json file')
    parser.add_argument('--height',
                        help='Image height: %(default)s',
                        metavar='<int>',
                        type=int,
                        default=768)
    parser.add_argument('-s',
                        '--switch',
                        help='Switch argument',
                        action='store_true')
    args = parser.parse_args()
    # lazy import
    from pathlib import Path
    import numpy as np
    from logzero import logger
    import pretrained_models
    # parser = pretrained_models.LitSMP.add_model_specific_args(parser)
    import transforms
    import mhd
    import point_utils

    import torch
    import cv2
    import img_utils
    from PIL import Image

    def load(filename, height):
        # x, _ = mhd.read(filename)
        img = Image.open(filename)
        x = np.array(img)
        if img.mode == 'I':
            x = x.astype(np.uint16)
        assert x.dtype == np.uint16, 'np.uint16 is expected'
        x = img_utils.resize_height(height, x)
        assert isinstance(x, np.ndarray)
        down_factor = 256
        new_shape = (down_factor *
                     np.ceil(np.array(x.shape) / down_factor)).astype(
                         np.int64).tolist()
        logger.info('Pad to %s', new_shape)
        x = img_utils.pad_to_shape(x, new_shape, 0)
        # x = transforms.apply_clahe(x, 100, (32, 32))
        return x

    logger.info('load model')
    lit_model = pretrained_models.LitSMP.load_from_checkpoint(args.model)
    lit_model.eval()
    logger.info('load image')
    x = load(args.input, args.height)
    original_shape = Image.open(args.input).size[::-1]

    def apply_clahe(img, clipLimit=100, tileGridSize=(32, 32)):
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        cl = clahe.apply(img)
        return cl

    x = apply_clahe(x, 5000, (32, 32))
    x = transforms.percentile_normalize(x, 1)  #, clip=True)
    if x.ndim == 2:
        x = x[np.newaxis]
    x = x.astype(np.float32)

    print(x.shape)
    if args.preproc:
        mhd.write(args.preproc, x)
    assert isinstance(x, np.ndarray)
    x = torch.from_numpy(x[np.newaxis])
    print(x.shape)
    pred = lit_model(x)
    print(pred.shape)
    pred = pred.to('cpu').detach().numpy().copy()
    print(pred.min(), pred.max())
    pred = np.round(np.clip(pred[0], 0, 1) * 255).astype(np.uint8)
    # print(kps_per_ch)
    if args.labelme:
        logger.info('Save %s', args.labelme)
        import json
        import labelme_utils
        kps_per_ch = point_utils.local_center_of_masses(pred)
        half_shape = (np.array(original_shape) // 2).tolist()
        scale = half_shape[0] / args.height
        kps_per_ch = [(np.array(kps, dtype=np.float32) * scale).tolist()
                      for kps in kps_per_ch]
        json_data = labelme_utils.kps2labelme(kps_per_ch,
                                              Path(args.input).name,
                                              half_shape,
                                              labelme_utils.LABEL_NAMES)
        # print(json_data)
        with open(args.labelme, 'w') as f:
            json.dump(json_data, f, indent=2)

    pred = np.transpose(pred, [1, 2, 0])
    mhd.write(args.output, pred, {'ElementNumberOfChannels': pred.shape[-1]})

    # keypoints, minmax, best_scores = h2p_affinity(pred)
    # arr = np.concatenate([keypoints, minmax, best_scores])
    # print(arr)
    # pred = np.transpose(pred, (1, 2, 0))  # channels to last
    # mhd.write(args.output, pred, {'ElementNumberOfChannels': pred.shape[-1]})


if __name__ == '__main__':
    sys.exit(main())