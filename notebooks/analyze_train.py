# %%
from pathlib import Path
import os
import json

from IPython.display import display
import pandas as pd
from IPython.core.display import HTML


def display_wrapped(tag: str, inner: str):
    wrapped = f'<{tag}>{inner}</{tag}>'
    display(HTML(wrapped))


result_dir = Path(os.environ.get('RESULT_DIR', '/poseidon/users/suzuki/projects/scoliosis/result/exp20241226'))
image_type = os.environ.get('IMAGE_TYPE', 'ap_whole')

csv_filename = os.environ.get('CSV_FILENAME', result_dir / f'postprocess/train/measures_{image_type}.csv')
flags_ndjson_filename = os.environ.get('FLAGS_FILENAME', result_dir / 'manual_annotation.ndjson')


# %%
df = pd.read_csv(csv_filename, index_col='filename')
df.index = pd.Index(df.index.str.split('/').str[-1], name='filename')
del df['unit_of_length']
# %%
df_ext = df.copy()
with open(flags_ndjson_filename) as f:
    flags = [json.loads(line) for line in f]

d_flags = {Path(flag['filename']).name: flag['content']['flags'] for flag in flags}
d_group_ids = {Path(flag['filename']).name: flag['content']['group_id'] for flag in flags}
df_ext['PostOP'] = df.index.map(lambda x: '術後' in d_flags[x])
df_ext['Patient ID'] = df.index.map(lambda x: d_group_ids[x])
print('Number of samples:', len(df_ext))
print('Number of unique patients:', df_ext['Patient ID'].nunique())
# %%
from functools import partial


def extract_l_count(key, d_flags):
    flag = d_flags[key]
    for l in ['L4', 'L5', 'L6']:
        if l in flag:
            return l
    print(f'No L found in {key}')
    return None


df_ext['L count'] = df.index.map(partial(extract_l_count, d_flags=d_flags))

# %%
df_ext.to_csv(str(csv_filename).replace('.csv', '_ext.csv'))
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# %%
name = 'Training data'
display_wrapped('h3', name)
display(df.describe())
# absolute
display_wrapped('h3', f'|{name}|')
display(df.abs().describe())
