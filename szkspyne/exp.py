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
import logzero
import mhd
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
from torch.utils.data import DataLoader
from tqdm import tqdm


# torch.backends.cudnn.benchmark = True # due to random cropping


def main():
    parser = argparse.ArgumentParser(description='Train and Test.')
    parser.add_argument('--input',
                        help='Input ndjson filename: %(default)s',
                        default='script/manual_annotation.ndjson')
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
    parser.add_argument(
        '--accelerator',
        help='Accelerator argument for pl.Trainer. default: %(default)s',
        default='auto')
    parser.add_argument(
        '--devices',
        help='devices argument for pl.Trainer. default: %(default)s',
        default='auto')
    parser.add_argument(
        '--strategy',
        help='strategy argument for pl.Trainer. default: %(default)s',
        default='ddp_find_unused_parameters_true')
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
    parser.add_argument('--train_cropped',
                        help='Train for cropped images',
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
    DEBUG = args.debug

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

    NDJSON_FILENAME = args.input
    shutil.copy(NDJSON_FILENAME, outroot)
    exclude_flags = set(['除外', '側面右向き'])
    df = exp_utils.ndjson2csv(NDJSON_FILENAME, exclude_flags)
    RANDOM_STATE = 2022
    if DEBUG:
        pass
        # df = df.sample(frac=.1, random_state=RANDOM_STATE).reset_index(drop=True)
    def extract_pid(filepath: str):
        fp = Path(filepath)
        if filepath.startswith('SSS'):
            return fp.stem.split('-')[0]
        elif filepath.startswith('CLO'):
            return 'CLO' + fp.stem.split('_')[0]
        else:
            return Path(filepath).parts[1]
    df['pid'] = df['filepath'].map(lambda pid: extract_pid(pid))
    df_all = df
    logger.info('All data %d', len(df_all))
    pids = df_all['pid'].unique()
    logger.info('Unique pid count %d', len(pids))
    K_FOLD = args.fold
    df_all['set_id'] = -1
    kfold = KFold(n_splits=K_FOLD,
                  shuffle=True,
                  random_state=RANDOM_STATE)#None if args.debug else 2022)
    for set_id, (_, test_index) in enumerate(kfold.split(pids)):
        idset = set(pids[test_index])
        df_all.loc[df_all['pid'].map(lambda e: e in idset), 'set_id'] = set_id
    df_all.to_csv(outroot / f'dataset_{os.getpid()}.csv', index=False)

    USE_TEST_AS_VAL = args.holdout

    NUM_WORKERS = len(
        os.sched_getaffinity(0)) - 1  #2 if os.name == 'nt' else 0
    NUM_WORKERS = min(NUM_WORKERS, 8)
    # if NUM_WORKERS > 8:
    #     NUM_WORKERS = max(10, NUM_WORKERS // 2)
    logger.info('workers %d', NUM_WORKERS)
    EPOCHS = 1 if DEBUG else 128
    if DEBUG:
        logger.warning('In debug mode')
    BATCH_SIZE = args.batch_size
    HEIGHT = args.height  # 256 * 5
    SIGMA = args.sigma

    def train_test(test_id):
        logger.info('test set: %d', test_id)
        outdir = outroot / '{}'.format(test_id)
        outdir.mkdir(parents=True, exist_ok=True)
        train_ids = [i for i in range(K_FOLD) if i != test_id]
        if USE_TEST_AS_VAL:
            df_train = df_all[df_all['set_id'].map(lambda e: e in train_ids)]
            df_val = df_all[df_all['set_id'] == test_id]
        else:
            df_train = df_all[df_all['set_id'].map(lambda e: e in train_ids)]
            non_test_pids = df_train['pid'].unique()
            rng = np.random.RandomState(RANDOM_STATE)
            val_pids = set(
                rng.choice(non_test_pids,
                                 size=int(0.1 * len(non_test_pids)),
                                 replace=False))
            train_pids = set(pids) - val_pids
            print('sizes pids, train_pids, val_pids', len(pids), len(train_pids), len(val_pids))
            df_val = df_train[df_train['pid'].map(lambda e: e in val_pids)]
            df_train = df_train[df_train['pid'].map(lambda e: e in train_pids)]
        if True:  # create ndjson files for post processing
            val_pids = set(df_val['pid'])
            train_pids = set(df_train['pid'])
            non_test_pids = train_pids.union(val_pids)
            with open(NDJSON_FILENAME) as f:
                all_data = [json.loads(l) for l in f.readlines()]
            valid_imgs = set(df_all['filepath'])
            print('all raw data', len(all_data))
            all_data = list(
                filter(
                    lambda d: d['filepath'] in valid_imgs, all_data))
            print('all filtered data', len(all_data))
            # all_data = list(
            #     filter(
            #         lambda d: len(set(d['flags']).intersection(exclude_flags))
            #         == 0, all_data))
            for d in all_data:
                pid = extract_pid(d['filepath'])
                d['pid'] = pid
                if pid in val_pids:
                    d['set'] = 'val'
                elif pid in train_pids:
                    d['set'] = 'train'
                else:
                    d['set'] = 'test'

            def save_ndjson(pids, filename):
                save_data = filter(lambda d: d['pid'] in pids, all_data)
                with open(filename, 'w') as f:
                    for d in save_data:
                        f.write(json.dumps(d, ensure_ascii=False))
                        f.write('\n')

            save_ndjson(set(non_test_pids), outdir / 'train_val.ndjson')
            save_ndjson(set(pids) - set(non_test_pids), outdir / 'test.ndjson')
        df_test = df_all[df_all['set_id'] == test_id]
        assert df_test['pid'].nunique() == len(set(pids) - set(non_test_pids)), f'Size mismatch {df_test["pid"].nunique()} vs {len(set(pids) - set(non_test_pids))}'
        df_test.to_csv(outdir / 'testset.csv', index=False)
        assert isinstance(df_train, pd.DataFrame) and isinstance(
            df_val, pd.DataFrame) and isinstance(df_test, pd.DataFrame)
        logger.info('train, val, test = %d, %d, %d', len(df_train),
                    len(df_val), len(df_test))

        logger.info('Create model')
        n_chs = 1
        N_CLASSES = len(LABEL_NAMES)
        lit_model = pretrained_models.LitSMP(n_chs, N_CLASSES, args)
        down_factor = 256

        if test_id == 0:
            lit_model.summary([BATCH_SIZE, n_chs, 1024, 1024])
            import pickle
            with open(outdir / 'model_hparams.pickle', 'wb') as f:
                pickle.dump(lit_model.hparams, f)

        def load_dataset(df: pd.DataFrame, height: Union[int, None],
                         clahe: bool):
            '''
            load images and keypoints
            '''

            def load(row):
                x, kp = exp_utils.load_data(row, data_root, LABEL_NAMES)
                if height is not None:
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
        height = None if args.train_cropped else HEIGHT
        train_xs, train_kps = load_dataset(df_train, height, False)
        if not DEBUG:
            train_xs, train_kps = train_xs + train_xs, train_kps + train_kps  # TODO: remove
        logger.info('train size %d', len(train_xs))
        logger.info('load val set')
        val_xs, val_kps = load_dataset(df_val, height, True)
        logger.info('val size %d', len(val_xs))
        flip_pairs = FLIP_PAIRS

        loader_kwargs = dict(batch_size=BATCH_SIZE,
                             pin_memory=True,
                             num_workers=NUM_WORKERS,
                             persistent_workers=True,
                             prefetch_factor=2)
        dataset = point_utils.KeypointDataset(
            train_xs,
            train_kps,
            None,
            flip_pairs,
            transforms.random_crop_transform
            if args.train_cropped else transforms.random_transform,
            sigma=SIGMA)
        train_loader = DataLoader(
            dataset,
            shuffle=True,
            drop_last=True,
            # collate_fn=img_utils.CropCollate(p=args.crop,
            #                                  margin=32,
            #                                  multiple=down_factor),
            **loader_kwargs,
        )

        def base_kps_transform(
                img, kps,
                flip_pairs) -> Tuple[np.ndarray, point_utils.PointsList]:
            img, kps = transforms.base_transform(img, kps)
            return img, kps

        dataset = point_utils.KeypointDataset(
            val_xs,
            val_kps,
            None,
            flip_pairs,
            transforms.base_crop_transform
            if args.train_cropped else base_kps_transform,
            sigma=SIGMA)
        val_loader = DataLoader(dataset,
                                shuffle=False,
                                drop_last=True,
                                **loader_kwargs)

        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger
        early_stop_callback = EarlyStopping(monitor='val_loss',
                                            patience=args.patience,
                                            verbose=False,
                                            mode='min')
        mc = ModelCheckpoint(monitor='val_loss', dirpath=outdir / 'checkpoint')
        csv_logger = CSVLogger(str(outdir), name='logs')
        precision = 16 if args.accelerator != 'cpu' else 32
        trainer = pl.Trainer(accelerator=args.accelerator,
                             devices=args.devices,
                             precision=precision,
                             strategy=args.strategy,
                             max_epochs=EPOCHS,
                             logger=csv_logger,
                             log_every_n_steps=len(train_loader)//8,
                             callbacks=[early_stop_callback, mc])

        logger.info('Start training')
        trainer.fit(lit_model, train_loader, val_loader)
        logger.info('Finish training')

        del dataset, train_loader, val_loader, train_xs, val_xs  # release memory

        if os.getenv("LOCAL_RANK", '0') != '0':
            print('Exit because LOCAL_RANK != 0')
            return 1

        df_logs = pd.read_csv(csv_logger.experiment.metrics_file_path)
        try:
            if len(df_logs)>1:
                df_logs = df_logs.iloc[1:]
            exp_utils.plot_history(df_logs)
            plt.gcf().savefig(str(outdir / 'history.png'),
                            transparent=True,
                            bbox_inches='tight')
            plt.clf()
        except Exception as e:
            logger.error('Failed to plot history: %s', e)


        test_xs, test_kps = load_dataset(df_test, HEIGHT, True)
        logger.info('test size %d', len(test_xs))

        def test_transform(img) -> np.ndarray:
            img = transforms.base_transform(img)
            assert isinstance(img, np.ndarray)
            return img

        dataset = point_utils.ImageDataset(test_xs, test_transform)
        test_loader = DataLoader(dataset,
                                 shuffle=False,
                                 drop_last=False,
                                 **loader_kwargs)

        logger.info('Start test')
        # trainer.predict(ckpt_path='best') # reload best weight
        logger.info('Load weight:%s', mc.best_model_path)
        lit_model.load_from_checkpoint(mc.best_model_path)
        # create new trainer with `devices=1`
        trainer = pl.Trainer(accelerator=args.accelerator,
                             devices=1, #args.devices,
                             precision=precision)
                            #  logger=csv_logger,
                            #  log_every_n_steps=len(train_loader)//8,
                            #  callbacks=[early_stop_callback, mc])
        y_preds = trainer.predict(model=lit_model, dataloaders=test_loader, ckpt_path=None)
        # y_preds = []
        # with torch.no_grad():
        #     y_preds.append(trainer.predict(dataloaders=test_loader, ckpt_path=None))
        assert y_preds is not None
        from szkspyne.apply_model import PROB_MAX
        y_preds = [
            (PROB_MAX * y_pred).cpu().numpy().astype(np.int16)  # type: ignore
            for y_pred in y_preds
        ]
        print(len(y_preds))
        y_preds = np.concatenate(y_preds, axis=0)
        print(y_preds.shape)
        logger.info('End test')
        if USE_TEST_AS_VAL:
            torch.save(
                lit_model.model.to('cpu').state_dict(),
                outdir / 'best_weights.pth')
        y_preds = np.moveaxis(y_preds, 1, -1).copy()  # to channel last
        with open(outdir / 'test.ndjson') as f:
            test_data = [json.loads(l) for l in f.readlines()]
        for image_type in ['ap_whole', 'lateral_whole']:
            indices = []
            for i, data in enumerate(test_data):
                if data['image_type'] == image_type:
                    indices.append(i)
            y_preds_it = y_preds[indices].copy()
            mhd.write(outdir / f'y_pred_{image_type}.mha', y_preds_it,
                      {'ElementNumberOfChannels': y_preds_it.shape[-1]})
            it_data = [test_data[i] for i in indices]
            with open(outdir / f'test_{image_type}.ndjson', 'w') as f:
                for data in it_data:
                    f.write(json.dumps(data))
                    f.write('\n')
        logger.info('End saving results')


        del y_preds, test_loader, dataset, test_xs, test_kps

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
