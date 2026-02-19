# %%
import json
import os
import subprocess
from math import sqrt
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from scipy import stats

from szkspyne.utils import scolrs

# display(HTML("<style>.main-container { max-width:600 !important; }</style>"))
# display(HTML('<style>table.dataframe{overflow-x: scroll}</style>'))


sns.set_theme(context='notebook', style='whitegrid', font='serif')
# %%

data_dir = Path(os.getenv('DATA_DIR', '/poseidon/users/suzuki/projects/scoliosis/result/neck_manual_20250703'))
ref_filename = str(data_dir / 'postprocess/measure_test_reference.ndjson')
ai_filename = str(data_dir / 'postprocess/measure_test_ai.ndjson')
exlude_filename = data_dir / 'exclude.txt'

if exlude_filename.exists():
    with open(exlude_filename, 'r') as f:
        exclude = [line.strip() for line in f.readlines() if line.strip()]
    exclude = set(exclude)
else:
    exclude = None

EXTERNAL_TEST = os.environ.get('EXTERNAL_TEST', 'False').lower() in ('true', '1', 'yes')


def load_as_measures(filename: str, exclude = None) -> List[scolrs.MeasureResultLine]:
    lines = scolrs.MeasureResultLine.from_ndjson(filename)
    new_lines = []
    # strip parent directory form .filename
    for line in lines:
        if exclude and Path(line.filename).stem in exclude:
            continue
        line.filename = Path(line.filename).name
        new_lines.append(line)
    return new_lines


ref_measures = load_as_measures(ref_filename, exclude)
ai_measures = load_as_measures(ai_filename, exclude)

ref_measure_dict = {m.filename: m for m in ref_measures}
ai_measure_dict = {m.filename: m for m in ai_measures}

# %%
diff_lines = []
for ai_measure in ai_measures:
    ref_measure = ref_measure_dict[ai_measure.filename]
    diff = ref_measure.content.difference(ai_measure.content)
    diff_line = scolrs.MeasureResultLine(filename=ai_measure.filename, content=diff)
    diff_lines.append(diff_line)

# %%
df = scolrs.MeasureResultLine.to_dataframe(diff_lines)
# df.head()

# %%


ref_flat_measures = ref_measures#[scolrs.MeasureResultLine(filename=m.filename, content=m.content.flatten()) for m in ref_measures]
ai_flat_measures = ai_measures#[scolrs.MeasureResultLine(filename=m.filename, content=m.content.flatten()) for m in ai_measures]

df_ref = scolrs.MeasureResultLine.to_dataframe(ref_flat_measures)
df_ref.to_csv(data_dir / 'postprocess/measures_reference.csv')
df_ref = df_ref.apply(pd.to_numeric, errors='coerce')
df_ai = scolrs.MeasureResultLine.to_dataframe(ai_flat_measures)
df_ai.to_csv(data_dir / 'postprocess/measures_ai.csv')
df_ai = df_ai.apply(pd.to_numeric, errors='coerce')

#%%
# diff_flat_measures = [
#     scolrs.MeasureResultLine(filename=m.filename, content=m.content.flatten()) for m in diff_lines
# ]
diff_flat_measures = diff_lines
df_diff = scolrs.MeasureResultLine.to_dataframe(diff_flat_measures)
df_diff.to_csv(data_dir / 'postprocess/measures_diff.csv')


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

cases_output_dir = data_dir / 'postprocess/cases'
cases_output_dir.mkdir(parents=True, exist_ok=True)
ref_case_dir = data_dir / 'postprocess/reference/draw'
if EXTERNAL_TEST:
    ai_case_dir = data_dir / 'ai/postprocess/draw'
else:
    ai_case_dir = data_dir / 'auto/postprocess/draw'

for col in df_ref.columns:
    n_non_nan = df_ref[col].notna().sum()
    if n_non_nan == 0:
        # print(f'{col}: no data')
        continue
    df = pd.concat([df_ref[col], df_ai[col]], axis=1)
    df.columns = ['Reference', 'AI']
    df = df.dropna()

    display_header('h3', f'{col}: {len(df)} data points')

    # ax = sns.scatterplot(data=df, x='Reference', y='AI', alpha=0.5)
    g = sns.jointplot(data=df, x='Reference', y='AI', joint_kws={'alpha': 0.5})
    # draw diagonal line
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    # g.ax_joint.plot(lims, lims, '-r')

    g.ax_joint.axline(
        (df['Reference'].min(), df['Reference'].min()),
        slope=1,
        color='gray',
        linestyle='--',
        label='y = x',
        alpha=0.25,
    )

    plt.xlabel('Reference')
    plt.ylabel('AI')
    # plt.title(col)
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
    df_sorted = df.sort_values('abs error', ascending=False).head(n=5)
    display_header('h4', 'Worst 5')
    display(df_sorted)

    # create case catalog
    n_cases = 5
    case_filenames = []
    for i, row in df_sorted.iloc[:n_cases].iterrows():
        ref_case = ref_case_dir / row.name
        ai_case = ai_case_dir / row.name
        case_filenames.extend([ref_case.with_suffix('.html'), ai_case.with_suffix('.html')])
    title = f'{col}: {len(df)} data points, worst 5 cases (Reference vs AI)'
    outfile = cases_output_dir / f'{col}_worst_cases.html'
    args = ['scolrs', 'catalog', *case_filenames, outfile, '--title', title]
    args = [str(arg) for arg in args]
    subprocess.check_call(args)
