#%%
from szkspyne import labelme_utils
import json

ndjson_filename = '../script/group_id_dicom_manual_annotation.ndjson'

group_ids = {}
with open(ndjson_filename, 'r') as f:
    lines = f.readlines()
objects = [json.loads(line) for line in lines]
shape_lines = [labelme_utils.ShapeDictLine.model_validate(obj) for obj in objects]
group_ids = {obj['filename']: obj['content']['group_id'] for obj in objects}

print(len(shape_lines))
# %%
import pandas as pd
rows = []
for shape_line in shape_lines:
    filename = shape_line.filename
    group_id = group_ids[filename]
    post_op = '術後' in shape_line.content.flags
    ap = 'ap_whole' in shape_line.content.flags
    if not ap:
        continue
    # print(filename, group_id, post_op)
    rows.append((filename, group_id, post_op))
df = pd.DataFrame(rows, columns=['filename', 'group_id', 'post_op'])
df.head()

# %%
# make pairs of pre-op and post-op images
pairs = []
for group_id, df_group in df.groupby('group_id'):
    df_group = df_group.sort_values('filename')
    df_group.drop_duplicates(['post_op'], inplace=True)
    if len(df_group) == 2:
        df_group.sort_values('post_op', inplace=True)
        pre_op = df_group.iloc[0]
        post_op = df_group.iloc[1]
        pairs.append((pre_op['filename'], post_op['filename']))

print('pairs', len(pairs))

# %%
import os
print(os.getcwd())
df_pairs = pd.DataFrame(pairs, columns=['pre_op', 'post_op'])
df_pairs.to_csv('pre_post_pairs.csv', index=False)