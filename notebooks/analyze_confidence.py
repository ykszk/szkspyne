#%%
from pathlib import Path
import pandas as pd
import numpy as np

data_dir = Path("/poseidon/users/suzuki/projects/scoliosis/result/exp20241226/external_test_1536/")

image_type = "ap_whole"
# image_type = "lateral_whole"
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
from IPython.display import display
from IPython.core.display import HTML
from functools import partial
from scipy import stats

def display_tag(tag: str, inner: str):
    wrapped = f'<{tag}>{inner}</{tag}>'
    display(HTML(wrapped))

def header_tag(level: int, inner: str):
    display_tag(f'h{level}', inner)

h2_tag = partial(header_tag, 2)
h3_tag = partial(header_tag, 3)
h4_tag = partial(header_tag, 4)


for col in df_error.columns:
    if col not in df_confidence.columns:
        continue
    h2_tag(f"Analyzing: {col}")
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
    
    # Calculate regression line (simple linear regression)
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_pair["confidence"], df_pair["error"])
    x_vals = np.array(plt.gca().get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='red', linestyle='--', label='Regression Line')
    plt.legend()

    plt.grid()
    plt.show()

    def print_subset(name, n=5, high=True):
        if high:
            subset = df_pair.nlargest(n, name)
        else:
            subset = df_pair.nsmallest(n, name)
        h4_tag(f"{'High' if high else 'Low'} {name} samples for {col}:")
        display(subset)

    print_subset("confidence", n=5, high=True)
    print_subset("confidence", n=5, high=False)
    print_subset("error", n=5, high=True)
    print_subset("error", n=5, high=False)

    # cases with high regression error
    h3_tag(f"Regression Error Analysis for {col}:")
    
    # Predict errors based on confidence using the regression line
    df_pair["predicted_error"] = slope * df_pair["confidence"] + intercept
    
    # Calculate regression residuals (actual error - predicted error)
    df_pair["regression_residual"] = df_pair["error"] - df_pair["predicted_error"]
    
    # Display samples with high positive residuals (actual error much higher than predicted)
    high_residual_threshold = df_pair["regression_residual"].quantile(0.9)  # Top 10%
    high_residual_samples = df_pair[df_pair["regression_residual"] > high_residual_threshold].copy()
    high_residual_samples = high_residual_samples.sort_values("regression_residual", ascending=False)
    
    h4_tag(f"High Regression Error samples for {col} (residual > {high_residual_threshold:.2f}):")
    display(high_residual_samples.head(10))
    
    # Also show samples with unexpectedly low errors (negative residuals)
    low_residual_threshold = df_pair["regression_residual"].quantile(0.1)  # Bottom 10%
    low_residual_samples = df_pair[df_pair["regression_residual"] < low_residual_threshold].copy()
    low_residual_samples = low_residual_samples.sort_values("regression_residual", ascending=True)
    
    h4_tag(f"Unexpectedly Low Error samples for {col} (residual < {low_residual_threshold:.2f}):")
    display(low_residual_samples.head(10))
    
    # Plot residuals
    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_pair, x="confidence", y="regression_residual")
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel("Confidence")
    plt.ylabel("Regression Residual")
    plt.title(f"Residuals vs Confidence for {col}")
    plt.grid()
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=df_pair, x="regression_residual", bins=30)
    plt.xlabel("Regression Residual")
    plt.ylabel("Count")
    plt.title(f"Distribution of Residuals for {col}")
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Regression equation: error = {slope:.4f} * confidence + {intercept:.4f}")
    print(f"R-squared: {r_value**2:.4f}")
    print(f"Standard error: {std_err:.4f}")

    # break