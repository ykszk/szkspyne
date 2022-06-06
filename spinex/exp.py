import os
import sys
if os.name == 'posix':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
import argparse
import json

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import logzero
from logzero import logger
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from tqdm import tqdm

import pretrained_models, transforms, point_utils, img_utils, exp_utils
from settings import LABEL_NAMES

import mhd

# torch.backends.cudnn.benchmark = True # due to random cropping


def main():
    parser = argparse.ArgumentParser(description='Train and Test.')
    parser.add_argument('--output',
                        help='Output directory: %(default)s',
                        metavar='<dirname>',
                        default='result')
    parser.add_argument('--patience',
                        help='The patience for early stopping: %(default)s',
                        metavar='<n>',
                        type=int,
                        default=8)
    parser.add_argument('--batch_size',
                        help='Batch size: %(default)s',
                        metavar='<n>',
                        type=int,
                        default=4)
    parser.add_argument('--height',
                        help='Image height: %(default)s',
                        metavar='<int>',
                        type=int,
                        default=768)
    parser.add_argument('--gpus',
                        help='# of gpus. Use all available gpus by default.',
                        metavar='<int>',
                        type=int)
    parser.add_argument('--sigma',
                        help='Sigma: %(default)s',
                        metavar='<sigma>',
                        type=float,
                        default=10.0)
    parser.add_argument('--crop',
                        help='Cropping probability in collation: %(default)s',
                        metavar='<value>',
                        type=float,
                        default=.2)
    parser.add_argument('--fold',
                        help='The number of folds in k-fold CV: %(default)s',
                        metavar='<n>',
                        type=int,
                        default=5)
    parser.add_argument('--debug',
                        help='Run in debug mode',
                        action='store_true')
    parser.add_argument('--clahe',
                        help='Use CLAHE for image normalization',
                        action='store_true')
    parser.add_argument(
        '--holdout',
        help=
        'Perform hold-out cross validation. Test fold is used as validation fold. Use this to train a model for deployment',
        action='store_true')

    parser = pretrained_models.LitSMP.add_model_specific_args(parser)

    args = parser.parse_args()

    data_root = Path('../data/')
    outroot = Path(args.output)
    outroot.mkdir(parents=True, exist_ok=True)
    logzero.logfile(outroot / 'output.log')
    logger.info('Copy source code')
    src_outdir = outroot / 'src'
    src_outdir.mkdir(parents=True, exist_ok=True)
    for fn in Path(__file__).parent.glob('*.py'):
        shutil.copy(fn, src_outdir)
    with open(outroot / 'args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    dfs = []
    # for fn in ['script/verse.csv', 'script/spineweb.csv']:
    for fn in ['script/manual_annotation.csv', 'script/spineweb.csv']:
        df = pd.read_csv(fn)
        #        df['pid'] = df['filepath'].map(lambda pid: Path(pid).parts[-1])
        df['pid'] = df['filepath']  #.map(lambda pid: Path(pid).parts[-1])
        dfs.append(df)
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    logger.info('All data %d', len(df_all))
    pids = df_all['pid']
    K_FOLD = args.fold
    df_all['set_id'] = -1
    kfold = KFold(n_splits=K_FOLD,
                  shuffle=not args.debug,
                  random_state=None if args.debug else 202105)
    for set_id, (_, test_index) in enumerate(kfold.split(pids)):
        idset = set(pids[test_index])
        df_all.loc[test_index, 'set_id'] = set_id  # type: ignore

    df_all.to_csv(outroot / 'dataset.csv', index=False)

    USE_TEST_AS_VAL = args.holdout

    DEBUG = args.debug
    NUM_WORKERS = len(
        os.sched_getaffinity(0)) - 1  #2 if os.name == 'nt' else 0
    # if NUM_WORKERS > 10:
    #     NUM_WORKERS = NUM_WORKERS // 2
    logger.info('workers %d', NUM_WORKERS)
    EPOCHS = 2 if DEBUG else 128
    if DEBUG:
        logger.warning('In debug mode')
    if args.gpus is None:
        gpus = torch.cuda.device_count()
    else:
        gpus = args.gpus
    logger.info('gpus %d', gpus)
    accelerator = 'dp'
    BATCH_SIZE = args.batch_size
    if accelerator == 'dp':
        BATCH_SIZE *= max(gpus, 1)
    HEIGHT = args.height  # 256 * 5
    SIGMA = args.sigma

    def train_test(test_id):
        logger.info('test set: %d', test_id)
        outdir = outroot / '{}'.format(test_id)
        outdir.mkdir(parents=True, exist_ok=True)
        val_id = (test_id + 1) % K_FOLD
        train_ids = [i for i in range(K_FOLD) if i != test_id and i != val_id]
        if USE_TEST_AS_VAL:
            train_ids = train_ids + [val_id]
            df_train = df_all[df_all['set_id'].map(lambda e: e in train_ids)]
            df_val = df_all[df_all['set_id'] == test_id]
        else:
            df_train = df_all[df_all['set_id'].map(lambda e: e in train_ids)]
            df_val = df_all[df_all['set_id'] == val_id]
        df_test = df_all[df_all['set_id'] == test_id]
        df_test.to_csv(outdir / 'testset.csv', index=False)
        logger.info('train, val, test = %d, %d, %d', len(df_train),
                    len(df_val), len(df_test))

        logger.info('Create model')
        n_chs = 1
        N_CLASSES = 4
        lit_model = pretrained_models.LitSMP(n_chs, N_CLASSES, args)
        down_factor = 256

        if test_id == 0:
            lit_model.summary([BATCH_SIZE, n_chs, 1024, 1024])

        def load_dataset(df, height, clahe):
            '''
            load images and keypoints
            '''

            def load(row):
                x, kp = exp_utils.load_data(row, data_root, LABEL_NAMES)
                x, kp = img_utils.resize_height(height, x, kp)
                if clahe:
                    x = transforms.apply_clahe(x)
                return x, kp

            data = Parallel(n_jobs=NUM_WORKERS)(
                delayed(load)(row)
                for _, row in tqdm(df.iterrows(), total=len(df)))
            xs, kps = zip(*data)

            shapes = [x.shape for x in xs]
            max_shape = np.max(shapes, axis=0)

            new_shape = (down_factor *
                         np.ceil(np.array(max_shape) / down_factor)).astype(
                             np.int64).tolist()
            logger.info('Pad to %s', new_shape)
            xs = [img_utils.pad_to_shape(x, new_shape, 0) for x in xs]
            return xs, kps

        logger.info('load train set')
        train_xs, train_kps = load_dataset(df_train, HEIGHT, False)
        train_xs, train_kps = train_xs + train_xs + train_xs, train_kps + train_kps + train_kps  # TODO: remove
        logger.info('train size %d', len(train_xs))
        logger.info('load val set')
        val_xs, val_kps = load_dataset(df_val, HEIGHT, True)
        logger.info('val size %d', len(val_xs))

        loader_kwargs = dict(batch_size=BATCH_SIZE,
                             pin_memory=True,
                             num_workers=NUM_WORKERS,
                             prefetch_factor=4)
        dataset = point_utils.KeypointDataset(train_xs,
                                              train_kps,
                                              None,
                                              transforms.random_transform,
                                              sigma=SIGMA)
        train_loader = DataLoader(
            dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=img_utils.CropCollate(p=args.crop,
                                             margin=32,
                                             multiple=down_factor),
            **loader_kwargs,
        )

        dataset = point_utils.KeypointDataset(val_xs,
                                              val_kps,
                                              None,
                                              transforms.base_transform,
                                              sigma=SIGMA)
        val_loader = DataLoader(dataset,
                                shuffle=False,
                                drop_last=True,
                                **loader_kwargs)

        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger
        from pytorch_lightning.plugins import DDPPlugin
        early_stop_callback = EarlyStopping(monitor='val_loss',
                                            patience=args.patience,
                                            verbose=False,
                                            mode='min')
        mc = ModelCheckpoint(monitor='val_loss', dirpath=outdir / 'checkpoint')
        csv_logger = CSVLogger(str(outdir), name='logs')
        trainer = pl.Trainer(
            gpus=gpus,
            precision=16 if gpus > 0 else 32,  # mixed precision
            strategy=accelerator,
            plugins=DDPPlugin(find_unused_parameters=False)
            if accelerator == 'ddp' else None,
            max_epochs=EPOCHS,
            logger=csv_logger,
            log_every_n_steps=len(train_loader),
            callbacks=[early_stop_callback, mc])

        logger.info('Start training')
        trainer.fit(lit_model, train_loader, val_loader)
        logger.info('Finish training')
        df_logs = pd.read_csv(csv_logger.experiment.metrics_file_path)
        exp_utils.plot_history(df_logs)
        plt.gcf().savefig(outdir / 'history.png',
                          transparent=True,
                          bbox_inches='tight')

        del dataset, train_loader, val_loader  # release memory

        test_xs, test_kps = load_dataset(df_test, HEIGHT, True)
        # np.save(outdir / 'kps.npy', np.array(test_kps))
        mhd.write(outdir / 'x_test.mha', np.stack(test_xs))
        import pickle
        with open(outdir / 'kps.pickle', 'wb') as f:
            pickle.dump(test_kps, f)
        dataset = point_utils.ImageDataset(test_xs, transforms.base_transform)
        test_loader = DataLoader(dataset, shuffle=False, **loader_kwargs)

        logger.info('Start test')
        y_preds = trainer.predict(dataloaders=test_loader, ckpt_path='best')
        assert y_preds is not None
        y_preds = np.concatenate(
            [y_pred.cpu().numpy() for y_pred in y_preds],  # type: ignore
            axis=0)
        logger.info('End test')
        y_preds = np.moveaxis(y_preds, 1, -1)  # to channel last
        mhd.write(outdir / 'y_pred.mha', y_preds,
                  {'ElementNumberOfChannels': y_preds.shape[-1]})
        logger.info('End saving results')
        if USE_TEST_AS_VAL:
            torch.save(
                lit_model.model.to('cpu').state_dict(),
                outdir / 'checkpoint' / 'best.pth')

    for test_id in range(K_FOLD):
        train_test(test_id)
        if DEBUG:
            logger.info('Break')
            break
        if USE_TEST_AS_VAL:
            logger.info('Break because of USE_TEST_AS_VAL')
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
