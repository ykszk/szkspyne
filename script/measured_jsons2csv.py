#%%
import json
from pathlib import Path
import pandas as pd
data_root = Path('tally')
IMAGE_TYPES = ['ap_whole', 'lateral_whole']
#%%
flags_required = ['術後','左屈','右屈','除外','L4','L5','L6','先天性側彎']
flag_dir = Path('data')
flags = {it:{} for it in IMAGE_TYPES}
for image_type in IMAGE_TYPES:
    indir = flag_dir / image_type
    for fn in indir.glob('*.json'):
        with open(fn) as f:
            data = json.load(f)
        set_flags = set([k for k, v in data['flags'].items() if v])
        flags[image_type][fn.stem] = {f:f in set_flags for f in flags_required}
#%%

manual_dir = data_root / 'manual'
auto_dir = data_root / 'auto'
for image_type in IMAGE_TYPES:
    dfs = []
    for indir in [manual_dir, auto_dir]:
        rows = []
        if indir.name=='manual':
            prefix='true '
        else:
            prefix='pred '
        for fn in (indir/image_type).glob('*.json'):
            data = {'filename':fn.stem}
            if indir.name=='manual':
                data.update(flags[image_type][fn.stem])
            with open(fn) as f:
                data.update({prefix+k:v for k,v in json.load(f).items()})
            rows.append(data)
        df = pd.DataFrame(rows)
        df.set_index('filename', drop=True, inplace=True)
        df.to_csv(data_root/f'{indir.name}_{image_type}.csv')
        dfs.append(df)
    df_merged = pd.concat(dfs, axis=1)
    df_merged.sort_index(inplace=True)
    df_merged.to_csv(data_root/f'{image_type}.csv', encoding='cp932')
#%%
