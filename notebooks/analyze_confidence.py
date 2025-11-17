#%%
from pathlib import Path
import pandas as pd

data_dir = Path("/poseidon/users/suzuki/projects/scoliosis/result/exp20241226/external_test_1536/")

# image_type = "ap_whole"
image_type = "lateral_whole"
error_path = data_dir / "postprocess" / f"no-bend_measures_{image_type}_diff.csv"
df_error = pd.read_csv(error_path, index_col=0)
df_error.drop("unit_of_length", axis=1, inplace=True)
print(df_error.shape)
df_error.head()
# %%
confidence_path = data_dir / "auto" / f"confidence_{image_type}.csv"
df_confidence = pd.read_csv(confidence_path, index_col=0)
print(df_confidence.shape)
df_confidence.head()
# %%
import seaborn as sns
import matplotlib.pyplot as plt

for col in df_error.columns:
    if col not in df_confidence.columns:
        continue
    plt.figure(figsize=(6, 4))
    df_pair = pd.DataFrame({
        "confidence": pd.to_numeric(df_confidence[col], errors='coerce'),
        "error": pd.to_numeric(df_error[col], errors='coerce').abs(),
    }).dropna()
    sns.scatterplot(data=df_pair, x="confidence", y="error")
    plt.xlabel("Confidence")
    plt.ylabel("Error")
    # correlation coefficient
    corr = df_pair["confidence"].corr(df_pair["error"])
    plt.title(f"Confidence vs Error for {col} (corr={corr:.2f})")
    plt.grid()
    plt.show()
    # break