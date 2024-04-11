import os
import sys


if os.name == 'posix':
    import matplotlib

    matplotlib.use('Agg')
import argparse
import json
import shutil
from pathlib import Path
from typing import Tuple, Union

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
from joblib import Parallel, delayed
from logzero import logger
from settings import FLIP_PAIRS, LABEL_NAMES
from sklearn.model_selection import KFold
from szkmipy import mhd
from torch.utils.data import DataLoader
from tqdm import tqdm


torch.backends.cudnn.benchmark = True  # due to random cropping


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
        '--crop', help='Cropping probability in collation: %(default)s', metavar='<value>', type=float, default=0.2
    )
    parser.add_argument('--max-epochs', help='Max epochs: %(default)s', type=int, default=128)
    parser.add_argument('--debug', help='Run in debug mode', action='store_true')
    parser.add_argument('--train_cropped', help='Train for cropped images', action='store_true')
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
    NUM_WORKERS = min(NUM_WORKERS, 8)
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
    down_factor = 256

    height = None if args.train_cropped else HEIGHT
    if os.getenv("LOCAL_RANK", '0') == '0':
        lit_model.summary([BATCH_SIZE, n_chs, height, height])
        import pickle

        with open(outdir / 'model_hparams.pickle', 'wb') as f:
            pickle.dump(lit_model.hparams, f)

    logger.info('load train set')
    train_xs, train_kps = exp_utils.load_dataset(
        df_train, height, data_root, LABEL_NAMES, NUM_WORKERS, down_factor, args.clahe
    )
    logger.info('Pad to %s', train_xs[0].shape)

    assert train_kps[0][0].size != 0, 'no points in the dataset'
    if not DEBUG:
        train_xs, train_kps = train_xs + train_xs, train_kps + train_kps  # TODO: remove
    logger.info('train size %d', len(train_xs))
    logger.info('load val set')
    val_xs, val_kps = exp_utils.load_dataset(
        df_val, height, data_root, LABEL_NAMES, NUM_WORKERS, down_factor, args.clahe
    )
    logger.info('Pad to %s', val_xs[0].shape)
    logger.info('val size %d', len(val_xs))
    flip_pairs = FLIP_PAIRS

    loader_kwargs = dict(
        batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=2
    )
    dataset = point_utils.KeypointDataset(
        train_xs,
        train_kps,
        None,
        flip_pairs,
        transforms.random_crop_transform if args.train_cropped else transforms.random_transform,
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
        val_xs,
        val_kps,
        None,
        flip_pairs,
        transforms.base_crop_transform if args.train_cropped else transforms.base_kps_transform,
        sigma=SIGMA,
    )
    val_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_kwargs)

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=False, mode='min')
    mc = ModelCheckpoint(monitor='val_loss', dirpath=outdir / 'checkpoint')
    csv_logger = CSVLogger(str(outdir), name='logs')
    wandb_logger = WandbLogger(project='scoliosis', name='/'.join(outdir.parts[-2:]), config=args)
    precision = 16 if args.accelerator != 'cpu' else 32
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

    if os.getenv("LOCAL_RANK", '0') != '0':
        print('Exit because LOCAL_RANK != 0')
        return 0

    del dataset, train_loader, val_loader, train_xs, val_xs  # release memory

    df_logs = pd.read_csv(csv_logger.experiment.metrics_file_path)
    try:
        if len(df_logs) > 1:
            df_logs = df_logs.iloc[1:]
        exp_utils.plot_history(df_logs)
        plt.gcf().savefig(str(outdir / 'history.png'), transparent=True, bbox_inches='tight')
        plt.clf()
    except Exception as e:
        logger.error('Failed to plot history: %s', e)

    test_xs, test_kps = exp_utils.load_dataset(
        df_test, height, data_root, LABEL_NAMES, NUM_WORKERS, down_factor, args.clahe
    )
    logger.info('test size %d', len(test_xs))

    dataset = point_utils.ImageDataset(test_xs, transforms.test_transform)
    test_loader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_kwargs)
    # xs = next(iter(test_loader))
    # mhd.write('tmp_test_xs.mha', xs.cpu().numpy())
    logger.info('Load weight:%s', mc.best_model_path)
    # lit_model.load_from_checkpoint(mc.best_model_path)
    lit_model = pretrained_models.LitSMP.load_from_checkpoint(mc.best_model_path)
    print(type(lit_model))
    logger.info('Start test')

    # create new trainer with `devices=1`
    trainer = pl.Trainer(accelerator=args.accelerator, devices=1, precision=precision)  # args.devices,
    #  logger=csv_logger,
    #  log_every_n_steps=len(train_loader)//8,
    #  callbacks=[early_stop_callback, mc])
    y_preds = trainer.predict(model=lit_model, dataloaders=test_loader, ckpt_path=None)
    # y_preds = []
    # with torch.no_grad():
    #     y_preds.append(trainer.predict(dataloaders=test_loader, ckpt_path=None))
    assert y_preds is not None
    logger.info('minmax: %s, %s', y_preds[0].min(), y_preds[1].max())
    from szkspyne.settings import PROB_MAX

    # to channel last
    y_preds = [(PROB_MAX * y_pred).permute((0, 2, 3, 1)).type(torch.ShortTensor) for y_pred in y_preds]  # type: ignore
    logger.info('minmax after prob_max: %s, %s', y_preds[0].min(), y_preds[1].max())
    y_preds = torch.cat(y_preds, 0).cpu().numpy().astype(np.int16)
    # y_preds = np.concatenate(y_preds, axis=0)
    logger.info('End test')
    if args.save_best:
        torch.save(lit_model.model.to('cpu').state_dict(), outdir / 'best_weights.pth')
    # y_preds = np.moveaxis(y_preds, 1, -1).copy()  # to channel last
    test_data = labelme_utils.load_ndjson(outdir / 'test.ndjson')
    for image_type in ['ap_whole', 'lateral_whole']:
        logger.info('Save preds %s', image_type)
        indices = []
        for i, data in enumerate(test_data):
            if image_type in data['content']['flags']:
                indices.append(i)
        y_preds_it = y_preds[indices].copy()
        mhd.write(outdir / f'y_pred_{image_type}.mha', y_preds_it, {'ElementNumberOfChannels': y_preds_it.shape[-1]})
        it_data = [test_data[i] for i in indices]
        labelme_utils.save_ndjson(it_data, outdir / f'test_{image_type}.ndjson')
    logger.info('End saving results')

    return 0


if __name__ == "__main__":
    sys.exit(main())
