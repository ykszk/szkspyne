# %%
import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


parser = argparse.ArgumentParser(description='Exp output to csv.')
parser.add_argument('root', help='root directory with ap_whole and lateral_whole directory', metavar='<input>')
args = parser.parse_args()
data_root = Path(args.root)
IMAGE_TYPES = ['ap_whole', 'lateral_whole']
flags_required = ['術後', 'EOS', '左屈', '右屈', '除外', 'L4', 'L5', 'L6', '先天性側彎']


def merge_ndjsons(ndjson_filenames: List[Path], prefixes: List[str]):
    dfs = []
    for i_fn, (ndjson_fn, prefix) in enumerate(zip(ndjson_filenames, prefixes)):
        with open(ndjson_fn) as f:
            rows = [json.loads(line) for line in f]

        for i, json_data in enumerate(rows):
            data = {'filename': json_data.pop('filename')}
            set_flags = json_data.pop('flags', {})
            if i_fn == 0:
                set_flags = set([k for k, v in set_flags.items() if v])
                flags = {f: f in set_flags for f in flags_required}
                data.update(flags)
            data.update({prefix + k: v for k, v in json_data.items()})
            rows[i] = data  # update
        df = pd.DataFrame(rows)
        df.set_index('filename', drop=True, inplace=True)
        dfs.append(df)
    df_merged = pd.concat(dfs, axis=1)
    df_merged.sort_index(inplace=True)
    return df_merged


PREFIXES = ['true ', 'pred ']
for image_type in IMAGE_TYPES:
    ndjsons = []
    for subdir in ['reference', 'auto']:
        indir = data_root / subdir
        ndjsons.append(indir / f'measures_{image_type}.ndjson')
    df_merged = merge_ndjsons(ndjsons, PREFIXES)
    df_merged.to_csv(data_root / f'{image_type}.csv', encoding='cp932')

for suffix in ['_w_curve_indices', '_w_closest_indices']:
    suffixed_filename = data_root / 'auto' / f'measures_ap_whole{suffix}.ndjson'
    if not suffixed_filename.exists():
        continue
    df_merged = merge_ndjsons(
        [
            data_root / 'reference' / 'measures_ap_whole.ndjson',
            suffixed_filename,
        ],
        PREFIXES,
    )
    df_merged.to_csv(data_root / f'ap_whole{suffix}.csv', encoding='cp932')
