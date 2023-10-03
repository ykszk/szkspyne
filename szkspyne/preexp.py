import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import exp_utils
import logzero
import numpy as np
import pandas as pd
from logzero import logger
from sklearn.model_selection import KFold


def main():
    parser = argparse.ArgumentParser(description='Train and Test.')
    parser.add_argument('--output', help='Output directory', metavar='<dirname>', required=True)
    parser.add_argument(
        '--fold', help='The number of folds in k-fold CV: %(default)s', metavar='<n>', type=int, default=5
    )
    parser.add_argument('--debug', help='Run in debug mode', action='store_true')
    parser.add_argument('--ndjson', help='Specify input ndjson', default='script/manual_annotation.ndjson')
    parser.add_argument('--seed', help='random state for splitting', default=2022)
    parser.add_argument(
        '--holdout',
        help='Perform hold-out cross validation. Test fold is used as validation fold. Use this to train a model for deployment',
        action='store_true',
    )

    args = parser.parse_args()
    DEBUG = args.debug

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

    NDJSON_FILENAME = outroot / 'manual_annotation.ndjson'
    shutil.copy(args.ndjson, NDJSON_FILENAME)
    exclude_flags = set(['除外', '側面右向き'])
    df = exp_utils.ndjson2csv(NDJSON_FILENAME, exclude_flags)

    df['pid'] = df['group_id']
    # if True: # TODO: remove after debugging
    #     df = df[df['pid'].map(lambda e:e.startswith('OSC'))]
    df_all = df
    logger.info('All data %d', len(df_all))
    pids = df_all['pid'].unique()
    logger.info('Unique pid count %d', len(pids))
    K_FOLD = args.fold
    df_all['set_id'] = -1
    kfold = KFold(n_splits=K_FOLD, shuffle=True, random_state=args.seed)  # None if args.debug else 2022)
    for set_id, (_, test_index) in enumerate(kfold.split(pids)):
        idset = set(pids[test_index])
        df_all.loc[df_all['pid'].map(lambda e: e in idset), 'set_id'] = set_id
    df_all.to_csv(outroot / f'dataset.csv', index=False)

    USE_TEST_AS_VAL = args.holdout

    def train_test(test_id):
        logger.info('test set: %d', test_id)
        outdir = outroot / '{}'.format(test_id)
        outdir.mkdir(parents=True, exist_ok=True)
        train_ids = [i for i in range(K_FOLD) if i != test_id]
        if USE_TEST_AS_VAL:
            df_train = df_all[df_all['set_id'].map(lambda e: e in train_ids)]
            df_val = df_all[df_all['set_id'] == test_id]
            test_pids = set(df_val['pid'])
        else:
            df_train = df_all[df_all['set_id'].map(lambda e: e in train_ids)]
            non_test_pids = df_train['pid'].unique()
            test_pids = set(pids) - set(non_test_pids)
            rng = np.random.RandomState(args.seed)
            val_pids = set(rng.choice(non_test_pids, size=int(0.1 * len(non_test_pids)), replace=False))
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
            all_data = list(filter(lambda d: d['filepath'] in valid_imgs, all_data))
            print('all filtered data', len(all_data))
            # all_data = list(
            #     filter(
            #         lambda d: len(set(d['flags']).intersection(exclude_flags))
            #         == 0, all_data))
            for d in all_data:
                pid = d['group_id']
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
                        f.write(json.dumps(d, ensure_ascii=False, separators=(',', ':')))
                        f.write('\n')

            save_ndjson(set(non_test_pids), outdir / 'train_val.ndjson')
            save_ndjson(test_pids, outdir / 'test.ndjson')
        df_test = df_all[df_all['set_id'] == test_id]
        assert df_test['pid'].nunique() == len(
            test_pids
        ), f'Size mismatch {df_test["pid"].nunique()} vs {len(test_pids)}'
        assert (
            isinstance(df_train, pd.DataFrame)
            and isinstance(df_val, pd.DataFrame)
            and isinstance(df_test, pd.DataFrame)
        )
        logger.info('train, val, test = %d, %d, %d', len(df_train), len(df_val), len(df_test))
        for df_sub, name in ((df_train, 'train'), (df_val, 'val'), (df_test, 'test')):
            df_sub.to_csv(outdir / f'{name}set.csv', index=False)
            df_sub.to_pickle(outdir / f'{name}set.pkl')
        # df_train.to_csv(outdir / 'trainset.csv', index=False)
        # df_val.to_csv(outdir / 'valset.csv', index=False)
        # df_test.to_csv(outdir / 'testset.csv', index=False)

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
