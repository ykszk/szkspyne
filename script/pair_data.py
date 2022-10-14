#%%
import numpy as np
from datetime import datetime
import pandas as pd
from typing import List
from pathlib import Path
import json

data_dir = Path('../../result/exp20221009')

with open(data_dir / 'manual_annotation.ndjson') as f:
    all_data = [json.loads(l) for l in f]
exclude_flags = set(['除外', '側面右向き'])
all_data = list(
    filter(lambda d: len(exclude_flags.intersection(d['flags'])) == 0,
           all_data))
len(all_data)
# %%
ap_data = [d for d in all_data if d['image_type'] == 'ap_whole']
for d in ap_data:
    if '術後' not in d['flags']:
        d['flags'].append('術前')
    filepath = Path(d['filepath'])
    d['PatientID'] = filepath.parts[1]
    d['StudyDate'] = filepath.parts[2]
    d['SeriesID'] = int(filepath.stem)
    d.pop('image_type', None)
    d.pop('point', None)
    d.pop('rectangle', None)
    d.pop('imageWidth', None)
    d.pop('imageHeight', None)
print(len(ap_data))
#%% flexion pairs
df = pd.DataFrame(ap_data)
df.sort_values('filepath',
               inplace=True)  # sort by PatientID, StudyDate, SeriesID
required_flags = set(['術前', '左屈', '右屈'])
complete_pids = []
completes = []
for (pid, study_date), df_pid in df.groupby(['PatientID', 'StudyDate']):
    if len(df_pid) < 3:
        continue
    flags = set()
    for _, row in df_pid.iterrows():
        flags = flags.union(set(row['flags']))
    if required_flags <= flags:
        completes.append(df_pid)
        complete_pids.append((pid, study_date))
df_complete = pd.concat(completes, axis=0, ignore_index=True)


def flags2id(flgs: List[str]):
    if '左屈' in flgs:
        return 1
    if '右屈' in flgs:
        return 2
    # if '臥位' in flgs:
    #     return 3
    return 0


df_complete['pose_id'] = df_complete['flags'].map(lambda flgs: flags2id(flgs))
df_complete['gai'] = df_complete['flags'].map(lambda flgs: 0
                                              if '臥位' in flgs else 1)
df_complete.sort_values(['PatientID', 'pose_id', 'gai'], inplace=True)
df_complete.reset_index(drop=True, inplace=True)
df_complete.drop_duplicates(['PatientID', 'StudyDate', 'pose_id'],
                            inplace=True)
del df_complete['gai']
print(len(complete_pids))
df_complete.to_csv(data_dir / 'ap_flexion_completed_set.csv',
                   index=False,
                   encoding='utf8')
df_complete.head(10)

#%% pre-/post-op pairs
df = pd.DataFrame(
    [d for d in ap_data if '左屈' not in d['flags'] and '右屈' not in d['flags']])
required_flags = set(['術前', '術後'])
complete_pids = []
completes = []
import itertools
for pid, df_pid in df.groupby('PatientID'):
    if len(df_pid) < 2:
        continue
    flags = set()
    for _, row in df_pid.iterrows():
        flags = flags.union(set(row['flags']))
    if not required_flags.issubset(flags):
        continue
    if len(df_pid) > 2:
        pre_indices = df_pid[df_pid['flags'].map(lambda e: '術前' in e)].index
        post_indices = df_pid[df_pid['flags'].map(lambda e: '術後' in e)].index
        pairs = list(itertools.product(pre_indices, post_indices))
        diffs = []
        for pre_idx, post_idx in pairs:
            post, pre = datetime.strptime(df_pid.loc[post_idx, 'StudyDate'],
                                          '%Y%m%d'), datetime.strptime(
                                              df_pid.loc[pre_idx, 'StudyDate'],
                                              '%Y%m%d')
            d = post - pre
            diffs.append(d)
        # print(diffs, np.min(diffs))
        pre_idx, post_idx = pairs[np.argmin(diffs)]
        completes.append(df_pid.loc[[pre_idx, post_idx]])
    else:
        completes.append(df_pid)
    complete_pids.append(pid)
df_complete = pd.concat(completes, axis=0, ignore_index=True)
df_complete.sort_values(['PatientID', 'StudyDate'], inplace=True)
df_complete.to_csv(data_dir / 'ap_op_completed_set.csv',
                   index=False,
                   encoding='utf8')
df_complete.tail(40)