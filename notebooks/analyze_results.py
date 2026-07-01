# %%
from pathlib import Path
import os
import json

import pandas as pd
from IPython.core.display import HTML
from IPython.display import display
from logzero import logger
import sklearn.metrics


def display_wrapped(tag: str, inner: str):
    wrapped = f'<{tag}>{inner}</{tag}>'
    display(HTML(wrapped))


logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO').upper())

# %%

result_dir = Path(os.environ['RESULT_DIR'])

image_type = os.environ.get('IMAGE_TYPE', 'ap_whole')
against_closest = os.environ.get('AGAINST_CLOSEST', 'false').lower() == 'true'

display_wrapped('h1', f'Analyze results for {image_type}')
df_ref = pd.read_csv(result_dir / f'reference/no-bend_measures_{image_type}.csv', index_col='filename')
if against_closest:
    df_ai = pd.read_csv(result_dir / f'closest/no-bend_measures_{image_type}.csv', index_col='filename')
    df_diff = pd.read_csv(result_dir / f'closest/no-bend_measures_{image_type}_diff.csv', index_col='filename')
else:
    df_ai = pd.read_csv(result_dir / f'auto/no-bend_measures_{image_type}.csv', index_col='filename')
    df_diff = pd.read_csv(result_dir / f'postprocess/no-bend_measures_{image_type}_diff.csv', index_col='filename')

# extract stem from index
for df in [df_ref, df_ai, df_diff]:
    df.index = pd.Index(df.index.str.split('/').str[-1], name='filename')


assert df_ref.index.equals(df_ai.index), 'Index is different between true and pred'
assert df_ref.index.equals(df_diff.index), 'Index is different between true and diff'

for df in [df_ref, df_ai, df_diff]:
    for col in ['unit_of_length']:
        # check if all values are the same
        assert df[col].nunique() == 1, f'{col} is different'
        # delete
        del df[col]

# convert columns to numeric
for df in [df_ref, df_ai, df_diff]:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


POST_OP = os.environ.get('POST_OP', None)
if POST_OP is not None:
    POST_OP = POST_OP.lower() == 'true'
    with open(result_dir / 'postprocess' / f'flags_{image_type}.ndjson') as f:
        flags = [json.loads(line) for line in f]

    selected = [
        Path(flag['filename']).name for flag in flags if flag['content']['flags'].get('PostOP', False) == POST_OP
    ]
    selected = [x for x in selected if x in df_ref.index]
    df_ref = df_ref.loc[selected].reindex().copy()
    df_ai = df_ai.loc[selected].reindex().copy()
    df_diff = df_diff.loc[selected].reindex().copy()


def load_flags(filename: Path) -> dict:
    with open(filename) as f:
        ndjson = [json.loads(line) for line in f]
    return {Path(obj['filename']).name: set([k for k, v in obj['content']['flags'].items() if v]) for obj in ndjson}


if True:
    if against_closest:
        output_dir = result_dir / 'closest'
    else:
        output_dir = result_dir / 'postprocess'
    d_flags = load_flags(result_dir / 'reference' / 'keypoints' / image_type / 'final_keypoints.ndjson')
    df_ref_ext = df_ref.copy()
    df_ref_ext['L count'] = df_ref_ext.index.map(lambda x: next(iter((d_flags[x] & {'L4', 'L5', 'L6'}))))

    d_flags = load_flags(result_dir / 'auto' / 'keypoints' / image_type / 'final_keypoints.ndjson')
    df_ai_ext = df_ai.copy()
    df_ai_ext['L count'] = df_ref_ext.index.map(lambda x: next(iter((d_flags[x] & {'L4', 'L5', 'L6'}))))

    d_flags = load_flags(result_dir / 'postprocess' / f'flags_{image_type}.ndjson')
    for df in [df_ref_ext, df_ai_ext]:
        df['PostOP'] = df.index.map(lambda x: 'PostOP' in d_flags[x])

    if POST_OP is None:
        df_ref_ext.to_csv(output_dir / f'no-bend_measures_{image_type}_reference_ext.csv')
        df_ai_ext.to_csv(output_dir / f'no-bend_measures_{image_type}_ai_ext.csv')

    count_labels = sorted(df_ref_ext['L count'].unique())
    df_cm = pd.DataFrame(
        sklearn.metrics.confusion_matrix(df_ref_ext['L count'], df_ai_ext['L count'], labels=count_labels),
        index=count_labels,
        columns=count_labels,
    )
    df_cm.index.name = 'Reference'
    df_cm.columns.name = 'AI'
    display_wrapped('h3', 'Confusion matrix')
    display(df_cm)


print('Number of samples:', len(df_ref))

group_id_files = [
    (result_dir / 'manual_annotation.ndjson'),
    (result_dir / 'holdout_group_id_dicom_manual_annotation.ndjson'),
]
group_ids = None
for group_id_file in group_id_files:
    if group_id_file.exists():
        with open(group_id_file) as f:
            ndjson = [json.loads(line) for line in f]
        group_ids = {Path(obj['filename']).name: obj['content']['group_id'] for obj in ndjson}
        break

if group_ids is None:
    raise FileNotFoundError('Manual annotation file not found')
group_ids = {k: v for k, v in group_ids.items() if k in df_ref.index}
print('Number of unique patients:', len(set(group_ids.values())))

# print(df_ref.shape, df_ai.shape, df_diff.shape)
df_ref.head()

# %%

from szkspyne.utils.stats import pearsonr_ci


def additional_desc(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df1.columns:
        nums = df1[col].notna() & df2[col].notna()
        r, p, lo, hi = pearsonr_ci(df1.loc[nums, col], df2.loc[nums, col])
        d = {'Pearson r': r}
        r2 = r**2
        d['R^2'] = r2
        rms = (df1.loc[nums, col] - df2.loc[nums, col]).pow(2).mean() ** 0.5
        d['RMS'] = rms
        mae = (df1.loc[nums, col] - df2.loc[nums, col]).abs().mean()
        d['MAE'] = mae
        rows.append(d)
    df_desc = pd.DataFrame(rows, index=df1.columns)
    return df_desc


for name, df in [('Reference', df_ref), ('AI', df_ai), ('Ref - AI', df_diff)]:
    # raw
    display_wrapped('h3', name)
    display(df.describe())
    # absolute
    display_wrapped('h3', f'|{name}|')
    display(df.abs().describe())


display_wrapped('h3', "Reference vs AI")
display(additional_desc(df_ref, df_ai).T)

# %%
import seaborn as sns
from matplotlib import pyplot as plt
import subprocess
import numpy as np

sns.set_theme(context="notebook", style="whitegrid")

display_wrapped('h2', 'Plots')
# create multi-indexed DataFrame
df_merged = pd.concat([df_ref, df_ai], axis=1)
df_merged.columns = pd.MultiIndex.from_product([['Reference', 'AI'], df_ref.columns])

cols = ['CobbPT', 'CobbMT', 'CobbTLL']
if image_type == 'lateral_whole':
    cols = [
        'T1ThoracicKyphosis',
        'ThoracicKyphosis',
        'T1Slope',
        'SagittalBalance',
        'PelvicIncidence',
        'PelvicTilt',
        'SacralSlope',
        'L5IncidenceAngle',
        'PelvicRadiusAngle',
        'LumbosacralAngle',
    ]

for col in cols:
    display_wrapped('h3', col)
    sns.scatterplot(data=df_merged, x=('Reference', col), y=('AI', col), alpha=0.25)
    plt.xlabel('Reference')
    plt.ylabel('AI')
    plt.show()

    df_sorted = df_diff[col].abs().sort_values(ascending=False).dropna()
    N = 5
    worst = (
        "Worst",
        slice(0, N),
    )
    first_q = (
        "First quartile",
        slice(1 * (len(df_sorted) // 4) - N // 2, int(1 * (len(df_sorted) // 4) + np.ceil(N / 2))),
    )
    second_q = (
        "Second quartile",
        slice(len(df_sorted) // 2 - N // 2, int(len(df_sorted) // 2 + np.ceil(N / 2))),
    )
    third_q = (
        "Third quartile",
        slice(3 * (len(df_sorted) // 4) - N // 2, int(3 * (len(df_sorted) // 4) + np.ceil(N / 2))),
    )
    # p_90 = (
    #     "90 percentile",
    #     slice(9 * (len(df_sorted) // 10) - N // 2, int(9 * (len(df_sorted) // 10) + np.ceil(N / 2))),
    # )
    # p_95 = (
    #     "95 percentile",
    #     slice(95 * (len(df_sorted) // 100) - N // 2, int(95 * (len(df_sorted) // 100) + np.ceil(N / 2))),
    # )
    p_5 = (
        "5 percentile",
        slice(
            int(5 * (len(df_sorted) / 100)) - N // 2,
            min(len(df_sorted), int(5 * (len(df_sorted) / 100) + np.ceil(N / 2))),
        ),
    )
    # best = (
    #     "Best",
    #     slice(-N, None),
    # )

    for name, sl in [worst, first_q, second_q, third_q, p_5]:
        # create side by side comparison
        ref_render_dir = result_dir / f'reference/{image_type}'
        if against_closest:
            ai_render_dir = result_dir / f'closest/{image_type}'
            side_by_side_dir = result_dir / 'closest/side_by_side'
        else:
            ai_render_dir = result_dir / f'auto/{image_type}'
            side_by_side_dir = result_dir / 'postprocess/side_by_side'
        if POST_OP is not None:
            side_by_side_dir = side_by_side_dir.with_name(side_by_side_dir.name + ('_PostOP' if POST_OP else '_PreOP'))
        side_by_side_dir.mkdir(exist_ok=True, parents=True)
        out_file = side_by_side_dir / f'{image_type}_{col}_{name}.html'
        filenames = []

        df_sub = pd.DataFrame(df_sorted[sl])
        if df_sub.empty:
            with open(out_file.with_suffix('.txt'), 'w') as f:
                f.write('None found')
            continue
        display_wrapped('h4', name)
        display(df_sub)

        for i, row in enumerate(df_sub.itertuples()):
            fn = str(row.Index)
            fn = fn[:-5]  # strip ".json"
            ref_file = ref_render_dir / f'{fn}.html'
            ai_file = ai_render_dir / f'{fn}.html'
            filenames.extend([ref_file, ai_file])
        title = f'{col} - {name} | First: reference, Second: AI'
        args = ['scolrs', 'catalog', *filenames, out_file, '--title', title]
        args = [str(arg) for arg in args]
        # print(' '.join(args))
        subprocess.check_output(args)
