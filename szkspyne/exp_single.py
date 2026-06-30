import os
import sys


if os.name == 'posix':
    import matplotlib

    matplotlib.use('Agg')
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


torch.set_num_threads(1)
# import multiprocessing as mp
# mp.set_start_method('spawn')

import exp_utils
import img_utils
import labelme_utils
import logzero
import numpy as np
import pandas as pd
import point_utils
import pretrained_models
import torch
import transforms
from logzero import logger
from settings import FLIP_PAIRS, LABEL_NAMES
from szkmipy import mhd
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback


torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description='Train and Test a fold.')
    parser.add_argument('--input', help='Input fold directory: %(default)s', metavar='<dirname>', required=True)
    parser.add_argument(
        '--patience', help='The patience for early stopping: %(default)s', metavar='<n>', type=int, default=8
    )
    parser.add_argument('--batch_size', help='Batch size: %(default)s', metavar='<n>', type=int, default=4)
    parser.add_argument('--height', help='Image height: %(default)s', metavar='<int>', type=int, default=768)
    parser.add_argument(
        '--accelerator', help='Accelerator argument for pl.Trainer. default: %(default)s', default='auto'
    )
    parser.add_argument('--devices', help='devices argument for pl.Trainer. default: %(default)s', default='auto')
    parser.add_argument(
        '--strategy',
        help='strategy argument for pl.Trainer. default: %(default)s',
        default='ddp_find_unused_parameters_true',
    )
    parser.add_argument('--sigma', help='Sigma: %(default)s', metavar='<sigma>', type=float, default=10.0)
    parser.add_argument(
        '--crop_labels',
        help='Label names used for cropping. Cropped images will be added to the training datset if specified',
        nargs="*",
        type=str,
        default=[],
    )
    parser.add_argument('--max-epochs', help='Max epochs: %(default)s', type=int, default=128)
    parser.add_argument('--debug', help='Run in debug mode', action='store_true')
    parser.add_argument('--save_best', help='Save best weights', action='store_true')
    parser.add_argument('--clahe', help='Use CLAHE for image normalization', action='store_true')

    parser = pretrained_models.LitSMP.add_model_specific_args(parser)

    args = parser.parse_args()
    DEBUG = args.debug

    data_root = Path('../data/')
    indir = Path(args.input)
    outroot = Path(args.input)
    with open(outroot / 'args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    NUM_WORKERS = len(os.sched_getaffinity(0)) - 1  # 2 if os.name == 'nt' else 0
    NUM_WORKERS = min(NUM_WORKERS, 16)
    logger.info('workers %d', NUM_WORKERS)
    EPOCHS = 1 if DEBUG else args.max_epochs
    if DEBUG:
        logger.warning('In debug mode')
    BATCH_SIZE = args.batch_size
    HEIGHT = args.height  # 256 * 5
    SIGMA = args.sigma

    df_train = pd.read_pickle(indir / 'trainset.pkl')
    df_val = pd.read_pickle(indir / 'valset.pkl')
    df_test = pd.read_pickle(indir / 'testset.pkl')
    outdir = indir
    logger.info('train, val, test = %d, %d, %d', len(df_train), len(df_val), len(df_test))

    logger.info('Create model')
    n_chs = 1
    N_CLASSES = len(LABEL_NAMES)
    lit_model = pretrained_models.LitSMP(n_chs, N_CLASSES, args)
    down_factor = pretrained_models.DOWN_FACTOR

    LOCAL_RANK = os.getenv("LOCAL_RANK", '0')
    if LOCAL_RANK == '0':
        lit_model.summary([BATCH_SIZE, n_chs, HEIGHT, HEIGHT])
        import pickle

        logger.info(f'Save model {LOCAL_RANK}')
        with open(outdir / 'model_hparams.pickle', 'wb') as f:
            pickle.dump(lit_model.hparams, f)

    logger.info('load train set')
    train_xs_kps = exp_utils.create_image_kps_loader(
        df_train, HEIGHT, data_root, LABEL_NAMES, NUM_WORKERS, down_factor, args.clahe, args.crop_labels
    )
    logger.info('Pad to %s', train_xs_kps.shape)

    train_kps = train_xs_kps[0][1]
    assert train_kps[0].size != 0, 'no points in the dataset'
    # if not DEBUG:
    #     train_xs, train_kps = train_xs + train_xs, train_kps + train_kps  # TODO: remove
    logger.info('train size %d', len(train_xs_kps))
    logger.info('load val set')
    val_xs_kps = exp_utils.create_image_kps_loader(
        df_val, HEIGHT, data_root, LABEL_NAMES, NUM_WORKERS, down_factor, args.clahe, args.crop_labels
    )
    logger.info('Pad to %s', val_xs_kps.shape)
    logger.info('val size %d', len(val_xs_kps))
    flip_pairs = FLIP_PAIRS

    loader_kwargs = dict(
        batch_size=BATCH_SIZE, pin_memory=False, num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2
    )
    dataset = point_utils.KeypointDataset(
        train_xs_kps,
        None,
        flip_pairs,
        transforms.random_transform,
        sigma=SIGMA,
    )
    train_loader = DataLoader(
        dataset,
        shuffle=True,
        drop_last=True,
        # collate_fn=img_utils.CropCollate(p=args.crop,
        #                                  margin=32,
        #                                  multiple=down_factor),
        **loader_kwargs,
    )

    dataset = point_utils.KeypointDataset(
        val_xs_kps,
        None,
        flip_pairs,
        transforms.base_kps_transform,
        sigma=SIGMA,
    )
    val_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_kwargs)

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=False, mode='min')
    mc = ModelCheckpoint(monitor='val_loss', dirpath=outdir / 'checkpoint')
    csv_logger = CSVLogger(str(outdir), name='logs')
    wandb_logger = WandbLogger(
        project='scoliosis', name='/'.join(outdir.parts[-2:]), save_dir=str(outdir), config=args
    )
    precision = '16-mixed' if args.accelerator != 'cpu' else 32
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=precision,
        strategy=args.strategy,
        max_epochs=EPOCHS,
        logger=[wandb_logger, csv_logger],
        log_every_n_steps=len(train_loader) // 8,
        callbacks=[early_stop_callback, mc],
    )

    logger.info('Compile model')
    torch.compile(lit_model, mode="reduce-overhead")
    logger.info('Start training')
    trainer.fit(lit_model, train_loader, val_loader)
    logger.info('Finish training')

    if LOCAL_RANK != '0':
        print('Exit because LOCAL_RANK != 0')
        return 0

    del dataset, train_loader, val_loader  # release memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    df_logs = pd.read_csv(csv_logger.experiment.metrics_file_path)
    try:
        if len(df_logs) > 1:
            df_logs = df_logs.iloc[1:]
        exp_utils.plot_history(df_logs)
        plt.gcf().savefig(str(outdir / 'history.png'), transparent=True, bbox_inches='tight')
        plt.clf()
    except Exception as e:
        logger.error('Failed to plot history: %s', e)

    height_multiplier = 1.0
    height_test = int(height_multiplier * HEIGHT)
    logger.info('load test set with height %d', height_test)
    test_xs, test_kps = exp_utils.load_dataset(
        df_test, height_test, data_root, LABEL_NAMES, NUM_WORKERS, down_factor, args.clahe
    )
    y_shape = test_xs[0].shape
    n_test = len(test_xs)
    logger.info('test size %d', n_test)
    test_xs = exp_utils.CompressedNumpyArray.build(test_xs)

    dataset = point_utils.ImageDataset(test_xs, transforms.test_transform)
    test_loader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_kwargs)
    logger.info('Load weight:%s', mc.best_model_path)
    with torch.serialization.safe_globals([argparse.Namespace]): # couldn't load without this line
        lit_model = pretrained_models.LitSMP.load_from_checkpoint(mc.best_model_path)
    print(type(lit_model))
    logger.info('Start test')

    class SaveCallback(Callback):
        def __init__(self, writer: mhd.Writer, func=None):
            self.writer = writer
            if func:
                self.func = func
            else:
                self.func = lambda x: x

        def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            for o in self.func(outputs):
                self.writer.write(o)

        def on_predict_end(self, trainer, pl_module):
            self.writer.close()

    from settings import PROB_MAX
    y_preds_shape = [n_test] + list(y_shape) + [N_CLASSES]
    header =  mhd.create_header(y_preds_shape, compress=True, last_dim_is_channel=True, compression_type='zstandard')
    writer = mhd.Writer(outdir / 'y_pred.mha', header, dtype=np.dtype('i2'))
    callback = SaveCallback(writer, lambda x: (PROB_MAX * x).permute((0, 2, 3, 1)).type(torch.ShortTensor).cpu().numpy())

    # create new trainer with `devices=1`
    trainer = pl.Trainer(accelerator=args.accelerator, devices=1, precision=precision, callbacks=[callback])
    trainer.predict(model=lit_model, dataloaders=test_loader, ckpt_path=None, return_predictions=False)
    logger.info('End test')

    if args.save_best:
        torch.save(lit_model.model.to('cpu').state_dict(), outdir / 'best_weights.pth')

    logger.info('End saving results')

    return 0


if __name__ == "__main__":
    sys.exit(main())
