# %%
from pathlib import Path
import json
import subprocess
import os

from IPython.display import display
from IPython.core.display import HTML


def display_wrapped(tag: str, inner: str):
    wrapped = f'<{tag}>{inner}</{tag}>'
    display(HTML(wrapped))


def header(text, level=3):
    display_wrapped(f'h{level}', text)


result_dir = os.environ.get(
    'RESULT_DIR', '/poseidon/users/suzuki/projects/scoliosis/result/implant20250116/external_test_screw'
)
result_dir = Path(result_dir)
eval_dir = result_dir / 'eval'
eval_dir.mkdir(exist_ok=True, parents=True)


def load_ndjson_as_dict(filename):
    with open(filename) as f:
        ndjson = [json.loads(line) for line in f]
    return {d['filename']: d['content'] for d in ndjson}


ref_data = load_ndjson_as_dict(result_dir / 'reference' / 'lm_implant_keypoints.ndjson')
ai_data = load_ndjson_as_dict(result_dir / 'ai' / 'dtron2_lm_combined.ndjson')

ref_counts = load_ndjson_as_dict(result_dir / 'reference' / 'counts.ndjson')
ai_counts = load_ndjson_as_dict(result_dir / 'ai' / 'counts.ndjson')
assert set(ref_counts.keys()) == set(ai_counts.keys())
# for content in ref_counts.values():
#     print(content)
#     break
# %%

# %%
import pandas as pd

pd.options.display.max_columns = None

df_ref = pd.DataFrame.from_dict(ref_counts, orient='index')
df_ai = pd.DataFrame.from_dict(ai_counts, orient='index')
for df in [df_ref, df_ai]:
    df['L4'] = df['L4'].fillna(0).astype(int)
    df['L5'] = df['L5'].fillna(0).astype(int)
    if 'L6' in df.columns:
        df['L6'] = df['L6'].fillna(0).astype(int)
    else:
        df['L6'] = 0
    arr = df.to_numpy()
    df['total'] = arr.sum(axis=1)
    # index of first non zero element in each row
    df['start'] = (arr != 0).argmax(axis=1)
    # index of last non zero element in each row
    df['end'] = 17 - (arr != 0)[:, ::-1].argmax(axis=1)
    # index of first non zero element in each row
    df['length'] = df['end'] - df['start'] + 1
    # index of first non zero element in each row
    df['density'] = arr.sum(axis=1) / df['length']

df_ref.to_csv(result_dir / 'eval' / 'reference_counts.csv')
header('Reference')
display(df_ref.head())
display(df_ref.describe().T)

# %%
import matplotlib.pyplot as plt

sum_ref = df_ref.abs().sum(axis=1)
sum_ref.hist(bins=30)
plt.xlabel('Sum of screw counts per image')


# %%
df_ai.to_csv(result_dir / 'eval' / 'ai_counts.csv')
header('AI')
display(df_ai.head())
display(df_ai.describe().T)


# %%
sum_ai = df_ai.abs().sum(axis=1)
sum_ai.hist(bins=30)
plt.xlabel('Sum of screw counts per image')


# %%
header('Reference - AI', level=2)
abs_diff_col = 'sum of  abs diffs'
df_diff = df_ref - df_ai
vertebral_columns = [f'T{i}' for i in range(1, 13)] + [f'L{i}' for i in range(1, 7)]
columns = vertebral_columns + ['total', 'start', 'end', 'length', 'density']
df_diff = df_diff[columns]
df_diff['sum of  abs diffs'] = df_diff.abs()[vertebral_columns].sum(axis=1)
df_diff.sort_values(abs_diff_col, ascending=False, inplace=True)
df_diff.to_csv(eval_dir / 'diff_counts.csv')
display(df_diff.head())
df_diff_desc = df_diff.describe()
# display(df_diff_desc.T)

# %%
stats_cols = ['total', 'length', 'density']
stats = {'root mean square error': [], 'mean abs error': [], 'pearons r': []}
for col in stats_cols:
    diff = df_diff[col]
    rms = (diff**2).mean() ** 0.5
    stats['root mean square error'].append(rms)
    mean_abs_error = diff.abs().mean()
    stats['mean abs error'].append(mean_abs_error)
    pearons_r = df_ref[col].corr(df_ai[col])
    stats['pearons r'].append(pearons_r)
df_stats = pd.DataFrame(stats, index=stats_cols).T
display(df_stats.T)
# %%

(sum_ref - sum_ai).hist(bins=30)
plt.xlabel('Difference of sum of screw counts per image')

# %%
print('Mean total absolute difference per image', df_diff[abs_diff_col].mean())
print('Mean absolute difference per vertebra', df_diff[df_diff.columns[:-1]].abs().mean().mean())

# %%

df_diff[abs_diff_col].hist(bins=100)
plt.xlabel('Sum of absolute errors per image')

# %%
import seaborn as sns

header('Error (Reference - AI) histogram per vertebra', level=2)

df_tidy = df_diff.reset_index().melt(
    id_vars='index', var_name='vertebra', value_name='error', value_vars=vertebral_columns
)
sns.histplot(data=df_tidy, hue='error', x='vertebra', bins=5, multiple="dodge", shrink=0.8)

# %%
from szkspyne.utils import doit

for name, sls in [
    ('worst cases', slice(None, 20)),
    ('median cases', slice(len(df_diff) // 2 - 10, len(df_diff) // 2 + 10)),
    ('best cases', slice(-10, None)),
]:
    header(name + ' in ' + abs_diff_col, level=2)
    display(df_diff.iloc[sls])
    cases = []
    for idx in df_diff.index[sls]:
        cases.append({'filename': idx, 'content': ref_data[idx]})
        cases.append({'filename': idx, 'content': ai_data[idx]})
    cases_ndjson = eval_dir / f'{name}.ndjson'
    with open(cases_ndjson, 'w') as f:
        for case in cases:
            json.dump(case, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
    title = f'{name} | First: reference, Second: AI'
    out_file = eval_dir / f'{name}.html'
    # --jobs 1 to keep the order of the images
    c1 = ['scolrs', 'svg-ndjson', cases_ndjson, '-', '--resize', '1600x1600', '--jobs', '1', 'coronal-implant']
    args = ['scolrs', 'catalog', '-', out_file, '--title', title]
    c2 = [str(arg) for arg in args]
    pipeline = doit.Pipeline().pipe(c1).pipe(c2)
    pipeline.run(text=True)

# %%
