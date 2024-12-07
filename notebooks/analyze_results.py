# %%
from pathlib import Path

import pandas as pd
from IPython.core.display import HTML
from IPython.display import display


def display_wrapped(tag: str, inner: str):
    wrapped = f'<{tag}>{inner}</{tag}>'
    display(HTML(wrapped))


result_dir = Path('/poseidon/users/suzuki/projects/scoliosis/result/exp20241113')

image_type = 'ap_whole'
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


sns.set_theme(context="notebook", style="whitegrid")

display_wrapped('h2', 'Plots')
# create multi-indexed DataFrame
df_merged = pd.concat([df_ref, df_ai], axis=1)
df_merged.columns = pd.MultiIndex.from_product([['Reference', 'AI'], df_ref.columns])

for col in ['CobbPT', 'CobbMT', 'CobbTLL']:
    display_wrapped('h3', col)
    sns.scatterplot(data=df_merged, x=('Reference', col), y=('AI', col))
    plt.xlabel('Reference')
    plt.ylabel('AI')
    plt.show()
