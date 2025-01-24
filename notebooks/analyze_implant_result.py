# %%
from pathlib import Path
import json
import subprocess

from IPython.display import display
from IPython.core.display import HTML


def display_wrapped(tag: str, inner: str):
    wrapped = f'<{tag}>{inner}</{tag}>'
    display(HTML(wrapped))


def header(text, level=3):
    display_wrapped(f'h{level}', text)


result_dir = Path('/poseidon/users/suzuki/projects/scoliosis/result/implant20250116/external_test')
eval_dir = result_dir / 'eval'
eval_dir.mkdir(exist_ok=True, parents=True)


def load_ndjson_as_dict(filename):
    with open(filename) as f:
        ndjson = [json.loads(line) for line in f]
    return {d['filename']: d['content'] for d in ndjson}


ref_counts = load_ndjson_as_dict(result_dir / 'reference' / 'counts.ndjson')
ai_counts = load_ndjson_as_dict(result_dir / 'ai' / 'counts.ndjson')
assert set(ref_counts.keys()) == set(ai_counts.keys())
# for content in ref_counts.values():
#     print(content)
#     break
# %%
import pandas as pd

df_ref = pd.DataFrame.from_dict(ref_counts, orient='index')
df_ai = pd.DataFrame.from_dict(ai_counts, orient='index')
for df in [df_ref, df_ai]:
    df['L5'] = df['L5'].fillna(0).astype(int)

df_ref.to_csv(result_dir / 'reference' / 'counts.csv')
header('Reference')
df_ref.head()

# %%
import matplotlib.pyplot as plt

sum_ref = df_ref.abs().sum(axis=1)
sum_ref.hist(bins=30)
plt.xlabel('Sum of screw counts per image')


# %%
df_ai.to_csv(result_dir / 'ai' / 'counts.csv')
header('AI')
df_ai.head()

# %%
sum_ai = df_ai.abs().sum(axis=1)
sum_ai.hist(bins=30)
plt.xlabel('Sum of screw counts per image')


# %%
header('Reference - AI', level=2)
df_diff = df_ref - df_ai
df_diff['Abs total'] = df_diff.abs().sum(axis=1)
df_diff.sort_values('Abs total', ascending=False, inplace=True)
df_diff.to_csv(eval_dir / 'diff_counts.csv')
df_diff.head(n=20)

# %%

(sum_ref - sum_ai).hist(bins=30)
plt.xlabel('Difference of sum of screw counts per image')

# %%
print('Mean total absolute difference per image', df_diff['Abs total'].mean())
print('Mean absolute difference per vertebra', df_diff[df_diff.columns[:-1]].abs().mean().mean())


# %%

df_diff['Abs total'].hist(bins=100)
plt.xlabel('Sum of absolute errors per image')

# %%
import seaborn as sns

header('Error (Reference - AI) histogram per vertebra', level=2)

value_vars = list(df_diff.columns[:-1])
df_tidy = df_diff.reset_index().melt(id_vars='index', var_name='vertebra', value_name='error', value_vars=value_vars)
sns.histplot(data=df_tidy, hue='error', x='vertebra', bins=5, multiple="dodge", shrink=0.8)

# %%

filenames = []
for idx in df_diff.index[:20]:
    name = f'{idx[:-5]}.svg'
    fn = result_dir / 'reference/svg' / name
    filenames.append(fn)
    fn = result_dir / 'ai/svg' / name
    filenames.append(fn)
title = f'Worst cases | First: reference, Second: AI'
out_file = eval_dir / 'worst_cases.html'
args = ['scolrs', 'catalog', *filenames, out_file, '--title', title]
args = [str(arg) for arg in args]
subprocess.check_output(args)

# %%
