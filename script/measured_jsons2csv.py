#%%
import json
from pathlib import Path
import pandas as pd
data_root = Path('../../result/exp20221009')
IMAGE_TYPES = ['ap_whole', 'lateral_whole']
#%%
flags_required = ['術後','左屈','右屈','除外','L4','L5','L6','先天性側彎']
# flag_dir = Path('/data/scoliosis/Osaka AI Team Dropbox/project/suzuki/scoliosis/annotation/completed')
# flags = {it:{} for it in IMAGE_TYPES}
# for image_type in IMAGE_TYPES:
#     indir = flag_dir / image_type
#     for fn in indir.glob('*.json'):
#         with open(fn) as f:
#             data = json.load(f)
#         set_flags = set([k for k, v in data['flags'].items() if v])
#         flags[image_type][fn.stem] = {f:f in set_flags for f in flags_required}
#%%

manual_dir = data_root / 'reference'
auto_dir = data_root / 'auto'
for image_type in IMAGE_TYPES:
    dfs = []
    for prefix, subdir in [('true ', 'reference'), ('pred ', 'auto')]:
        indir = data_root/subdir
        rows = []
        for fn in (indir/image_type).glob('*.json'):
            data = {'filename':fn.stem}
            # if subdir=='reference':
            #     data.update(flags[image_type][fn.stem])
            with open(fn) as f:
                json_data = json.load(f)
                set_flags = json_data.pop('flags', {})
                set_flags = set([k for k, v in set_flags.items() if v])
                flags = {f:f in set_flags for f in flags_required}
                if subdir=='reference':
                    data.update(flags)
                data.update({prefix+k:v for k,v in json_data.items()})
            rows.append(data)
        df = pd.DataFrame(rows)
        df.set_index('filename', drop=True, inplace=True)
        # df.to_csv(data_root/f'{subdir}_{image_type}.csv')
        dfs.append(df)
    df_merged = pd.concat(dfs, axis=1)
    df_merged.sort_index(inplace=True)
    df_merged.to_csv(data_root/f'{image_type}.csv', encoding='cp932')
#%%
