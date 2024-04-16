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
    parser = argparse.ArgumentParser(description='Prepare dataset for experiments.')
    parser.add_argument('input', help='Input ndjson in ShapeDictLine format')
    parser.add_argument('--output', help='Output directory', metavar='<dirname>', required=True)
    parser.add_argument(
        '--fold', help='The number of folds in k-fold CV: %(default)s', metavar='<n>', type=int, default=5
    )
    parser.add_argument('--val_ratio', help='Validation ratio: %(default)s', type=float, default=0.1)
    parser.add_argument('--seed', help='random state for splitting', default=2022)
    parser.add_argument(
        '--holdout',
        help='Perform hold-out validation instead of k-fold cross-validation.',
        action='store_true',
    )

    args = parser.parse_args()

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

    output_ndjson_filename = outroot / 'manual_annotation.ndjson' # filename has to be this because post processing uses this name
    shutil.copy(args.input, output_ndjson_filename)
    df_all = exp_utils.ndjson2df(output_ndjson_filename)

    logger.info('All data %d', len(df_all))
    group_ids = df_all['group_id'].unique()
    logger.info('Unique group_id count %d', len(group_ids))
    K_FOLD = args.fold
    df_all['set_id'] = -1
    kfold = KFold(n_splits=K_FOLD, shuffle=True, random_state=args.seed)
    for set_id, (_, test_index) in enumerate(kfold.split(group_ids)):
        idset = set(group_ids[test_index])
        df_all.loc[df_all['group_id'].isin(idset), 'set_id'] = set_id
    df_all.to_csv(outroot / f'dataset.csv', index=False)

    def train_test(test_id):
        logger.info('test set: %d', test_id)
        outdir = outroot / 'folds' / '{}'.format(test_id)
        outdir.mkdir(parents=True, exist_ok=True)
        train_ids = [i for i in range(K_FOLD) if i != test_id]
        if args.holdout:
            df_train = df_all[df_all['set_id'].isin(train_ids)]
            df_val = df_all[df_all['set_id'] == test_id]
            test_group_ids = set(df_val['group_id'])
        else:
            df_train = df_all[df_all['set_id'].isin(train_ids)]
            non_test_group_ids = df_train['group_id'].unique()
            test_group_ids = set(group_ids) - set(non_test_group_ids)
            rng = np.random.RandomState(args.seed)
            val_group_ids = set(rng.choice(non_test_group_ids, size=int(args.val_ratio * len(non_test_group_ids)), replace=False))
            train_group_ids = set(group_ids) - val_group_ids
            print('sizes group_ids, train_group_id, val_group_id', len(group_ids), len(train_group_ids), len(val_group_ids))
            df_val = df_train[df_train['group_id'].isin(val_group_ids)]
            df_train = df_train[df_train['group_id'].isin(train_group_ids)]
        if True:  # create ndjson files for post processing
            val_group_ids = set(df_val['group_id'])
            train_group_ids = set(df_train['group_id'])
            non_test_group_ids = train_group_ids.union(val_group_ids)
            with open(output_ndjson_filename) as f:
                all_data = [json.loads(l) for l in f.readlines()]
            valid_imgs = set(df_all['imagePath'])
            print('all raw data', len(all_data))
            all_data = list(filter(lambda d: d['content']['imagePath'] in valid_imgs, all_data))
            print('all filtered data', len(all_data))

            for d in all_data:
                group_id = d['content']['group_id']
                if group_id in val_group_ids:
                    d['set'] = 'val'
                elif group_id in train_group_ids:
                    d['set'] = 'train'
                else:
                    d['set'] = 'test'

            def save_ndjson(group_ids, filename):
                save_data = filter(lambda d: d['content']['group_id'] in group_ids, all_data)
                with open(filename, 'w') as f:
                    for d in save_data:
                        f.write(json.dumps(d, ensure_ascii=False, separators=(',', ':')))
                        f.write('\n')

            save_ndjson(set(non_test_group_ids), outdir / 'train_val.ndjson')
            save_ndjson(test_group_ids, outdir / 'test.ndjson')
        df_test = df_all[df_all['set_id'] == test_id]
        assert df_test['group_id'].nunique() == len(
            test_group_ids
        ), f'Size mismatch {df_test["group_id"].nunique()} vs {len(test_group_ids)}'
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
        if args.holdout:
            logger.info('Break because of USE_TEST_AS_VAL')
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
