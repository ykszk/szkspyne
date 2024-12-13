# %%
from pathlib import Path
import os

import pandas as pd
from IPython.core.display import HTML
from IPython.display import display


def display_wrapped(tag: str, inner: str):
    wrapped = f'<{tag}>{inner}</{tag}>'
    display(HTML(wrapped))


result_dir = Path(os.environ.get('RESULT_DIR', '/poseidon/users/suzuki/projects/scoliosis/result/exp20241113'))

image_type = os.environ.get('IMAGE_TYPE', 'ap_whole')

display_wrapped('h1', f'Analyze results for {image_type}')
df_ref = pd.read_csv(result_dir / f'reference/no-bend_measures_{image_type}.csv', index_col='filename')
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
        'ThoracicKyphosis',
        'ProximalThoracicKyphosis',
        'MidLowerThoracicKyphosis',
        'LumbarLordosis',
    ]

for col in cols:
    display_wrapped('h3', col)
    sns.scatterplot(data=df_merged, x=('Reference', col), y=('AI', col))
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
            5 * (len(df_sorted) // 100) - N // 2,
            min(len(df_sorted), int(5 * (len(df_sorted) // 100) + np.ceil(N / 2))),
        ),
    )
    # best = (
    #     "Best",
    #     slice(-N, None),
    # )

    for name, sl in [worst, first_q, second_q, third_q, p_5]:
        df_sub = pd.DataFrame(df_sorted[sl])
        display_wrapped('h4', name)
        display(df_sub)

        # create side by side comparison
        ref_render_dir = result_dir / f'reference/{image_type}'
        ai_render_dir = result_dir / f'auto/{image_type}'
        side_by_side_dir = result_dir / f'postprocess/side_by_side'
        side_by_side_dir.mkdir(exist_ok=True, parents=True)
        filenames = []
        for i, row in enumerate(df_sub.itertuples()):
            fn = str(row.Index)
            fn = fn[:-5]  # strip ".json"
            ref_file = ref_render_dir / f'{fn}.html'
            ai_file = ai_render_dir / f'{fn}.html'
            filenames.extend([ref_file, ai_file])
        out_file = side_by_side_dir / f'{image_type}_{col}_{name}.html'
        title = f'{col} | First: reference, Second: AI'
        args = ['neckrs', 'catalog', *filenames, out_file, '--title', title]
        args = [str(arg) for arg in args]
        # print(' '.join(args))
        subprocess.check_output(args)
