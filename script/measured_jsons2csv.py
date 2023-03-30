#%%
import json
from pathlib import Path
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='Exp output to csv.')
parser.add_argument(
    'root',
    help='root directory with ap_whole and lateral_whole directory',
    metavar='<input>')
args = parser.parse_args()
data_root = Path(args.root)
IMAGE_TYPES = ['ap_whole', 'lateral_whole']
flags_required = ['術後', '左屈', '右屈', '除外', 'L4', 'L5', 'L6', '先天性側彎']

for image_type in IMAGE_TYPES:
    dfs = []
    for prefix, subdir in [('true ', 'reference'), ('pred ', 'auto')]:
        indir = data_root / subdir
        ndjson_fn = indir/f'measures_{image_type}.ndjson'
        if (ndjson_fn).exists():
            print('Use ndjson', ndjson_fn)
            with open(ndjson_fn) as f:
                rows = [json.loads(line) for line in f]
        else:
            rows = []
            for fn in (indir / image_type).glob('*.json'):
                with open(fn) as f:
                    json_data = json.load(f)
                    json_data['filename'] = fn.stem
                    rows.append(json_data)
        for i, json_data in enumerate(rows):
            data = {'filename': json_data.pop('filename')}
            set_flags = json_data.pop('flags', {})
            if subdir == 'reference':
                set_flags = set([k for k, v in set_flags.items() if v])
                flags = {f: f in set_flags for f in flags_required}
                data.update(flags)
            data.update({prefix + k: v for k, v in json_data.items()})
            rows[i] = data # update
        df = pd.DataFrame(rows)
        df.set_index('filename', drop=True, inplace=True)
        dfs.append(df)
    df_merged = pd.concat(dfs, axis=1)
    df_merged.sort_index(inplace=True)
    df_merged.to_csv(data_root / f'{image_type}.csv', encoding='cp932')
