# %% metadata for Pretty Jupyter [raw]
# title: Neck measurement result analysis

# %%
from typing import List
from szkspyne.utils import scolrs
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import sqrt
from IPython.display import display, HTML
import pandas as pd
import os

# display(HTML("<style>.main-container { max-width:600 !important; }</style>"))
# display(HTML('<style>table.dataframe{overflow-x: scroll}</style>'))


sns.set_theme(context='notebook', style='whitegrid', font='serif')
# %%

data_dir = Path(os.getenv('DATA_DIR', '/poseidon/users/suzuki/projects/scoliosis/result/neck_manual_20250217/folds/0'))
ref_filename = str(data_dir / 'postprocess/measure_test_reference.ndjson')
ai_filename = str(data_dir / 'postprocess/measure_test_ai.ndjson')


def load_as_measures(filename: str) -> List[scolrs.MeasureListResultLine]:
    lines = scolrs.MeasureListResultLine.from_ndjson(filename)
    # strip parent directory form .filename
    for line in lines:
        line.filename = Path(line.filename).name
    return lines


ref_measures = load_as_measures(ref_filename)
ai_measures = load_as_measures(ai_filename)

ref_measure_dict = {m.filename: m for m in ref_measures}
ai_measure_dict = {m.filename: m for m in ai_measures}

# %%
diff_lines = []
for ai_measure in ai_measures:
    ref_measure = ref_measure_dict[ai_measure.filename]
    diff = ref_measure.content.difference(ai_measure.content)
    diff_line = scolrs.MeasureListResultLine(filename=ai_measure.filename, content=diff)
    diff_lines.append(diff_line)

# %%
df = scolrs.MeasureListResultLine.to_dataframe(diff_lines)
# df.head()

# %%


ref_flat_measures = [scolrs.MeasureResultLine(filename=m.filename, content=m.content.flatten()) for m in ref_measures]
ai_flat_measures = [scolrs.MeasureResultLine(filename=m.filename, content=m.content.flatten()) for m in ai_measures]

df_ref = scolrs.MeasureResultLine.to_dataframe(ref_flat_measures)
df_ref.to_csv(data_dir / 'measures_reference.csv')
df_ref = df_ref.apply(pd.to_numeric, errors='coerce')
df_ai = scolrs.MeasureResultLine.to_dataframe(ai_flat_measures)
df_ai.to_csv(data_dir / 'measures_ai.csv')
df_ai = df_ai.apply(pd.to_numeric, errors='coerce')

# %%
exclude_cols = ['unit_of_length', 'Sacs']
desc_cols = df_ref.columns.difference(exclude_cols)


def display_header(tag: str, inner: str):
    wrapped = f'<{tag}>{inner}</{tag}>'
    display(HTML(wrapped))


display_header('h2', 'Reference measure summary')
df_ref[desc_cols].describe()

# %%
display_header('h3', 'Reference: absolute values')
df_ref[desc_cols].abs().describe()

# %%
display_header('h2', 'AI measure summary')
df_ai[desc_cols].describe()

# %%
display_header('h3', 'AI: absolute values')
df_ai[desc_cols].abs().describe()

# %%


display_header('h2', 'Scatter plots')

for col in df_ref.columns:
    n_non_nan = df_ref[col].notna().sum()
    if n_non_nan == 0:
        # print(f'{col}: no data')
        continue
    df = pd.concat([df_ref[col], df_ai[col]], axis=1)
    df.columns = ['Reference', 'AI']
    df = df.dropna()

    display_header('h3', f'{col}: {len(df)} data points')
    plt.axline(
        (df['Reference'].min(), df['Reference'].min()),
        slope=1,
        color='gray',
        linestyle='--',
        label='y = x',
        alpha=0.25,
    )
    ax = sns.scatterplot(data=df, x='Reference', y='AI', alpha=0.5)
    # draw diagonal line
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]

    plt.xlabel('Reference')
    plt.ylabel('AI')
    plt.title(col)
    plt.legend()
    plt.show()

    r, p = stats.pearsonr(df['Reference'], df['AI'])
    r2 = r**2
    rms = sqrt(((df['Reference'] - df['AI']) ** 2).mean())
    mae = (df['Reference'] - df['AI']).abs().mean()
    series = pd.DataFrame([{'Pearson r': r, 'R^2': r2, 'RMS': rms, 'MAE': mae}])
    series.index = [col]
    display(series)

    df['error'] = df['AI'] - df['Reference']
    df['abs error'] = df['error'].abs()

    # worst 5
    display_header('h4', 'Worst 5')
    display(df.sort_values('abs error', ascending=False).head(n=5))
