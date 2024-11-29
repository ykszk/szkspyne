# %%
from pathlib import Path
import pandas as pd

result_dir = Path('/poseidon/users/suzuki/projects/scoliosis/result/exp20241113')
image_type = 'ap_whole'
df_ref = pd.read_csv(result_dir / f'reference/no-bend_measures_{image_type}.csv', index_col='filename')
df_ai = pd.read_csv(result_dir / f'auto/no-bend_measures_{image_type}.csv', index_col='filename')
df_diff = pd.read_csv(result_dir / f'postprocess/no-bend_measures_{image_type}_diff.csv', index_col='filename')

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

print(df_ref.shape, df_ai.shape, df_diff.shape)
df_ref.head()

# %%
from IPython.core.display import display, HTML


def display_wrapped(tag: str, inner: str):
    wrapped = f'<{tag}>{inner}</{tag}>'
    display(HTML(wrapped))


for name, df in [('Reference', df_ref), ('AI', df_ai), ('Ref - AI', df_diff)]:
    # raw
    display_wrapped('h2', name)
    display(df.describe())
    # absolute
    display_wrapped('h3', f'|{name}|')
    display(df.abs().describe())
