# %%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

SEP = "=" * 20

# %%
# Load and print summary statistics
try:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent.__str__()
except Exception as e:
    proj_root = "/home/ksmith/birds/neural_networks/fraud_detection"

data_file = proj_root + "/data/creditcard.csv"
raw_df = pd.read_csv(data_file)

# %%
print(SEP)
print("DTYPES")
print(SEP)
print(raw_df.dtypes)

print(SEP)
print("COLUMNS")
print(SEP)
print(raw_df.columns)

print(SEP)
print("DATAFRAME DESCRIPTION")
print(SEP)
print(raw_df.describe())
raw_df.describe().to_csv(proj_root + "/data/raw_df_described.csv")
# print(raw_df.describe(include="object"))

# %%
print(SEP)
print("CLASS DISTRIBUTION")
print(SEP)
classes = raw_df.Class.unique()
print(f"Classes: {classes}")
for idx, val in enumerate(classes):
    class_count = len(raw_df[raw_df.Class == val])
    total_len = len(raw_df)
    print(f"Count of class {val}: {class_count:.2%} | {(class_count / total_len):.2%}")

# %%
print(SEP)
print("CLASS DESCRIPTIONS")
print(SEP)
class_0_df = raw_df[raw_df.Class == 0]
class_1_df = raw_df[raw_df.Class == 1]
class_0_described = class_0_df.describe()
class_1_described = class_1_df.describe()
print(class_0_df.describe())
print(class_1_df.describe())


# %%
# Variance between classes
data = {}
for col in class_0_described.columns:
    data[col] = class_0_described[col].values - class_1_described[col].values

class_difference_df = pd.DataFrame(
    index=class_0_described.index, columns=class_0_described.columns, data=data
)
class_difference_df.to_csv(proj_root + "/data/class_difference_df.csv")

# %%
# nrows = int(round(np.sqrt(class_0_described.shape[1]), 0))
# ncols = nrows
# plt.subplots(nrows, ncols)

fig, ax = plt.subplots(figsize=(14, 6))

# Extract the mean values
means_0 = class_0_described.loc["mean", "V1":"V28"]
means_1 = class_1_described.loc["mean", "V1":"V28"]

# Create x positions
x = np.arange(len(means_0))
width = 0.35

# Plot grouped bars
ax.bar(x - width / 2, means_0, width, label="Legitimate")
ax.bar(x + width / 2, means_1, width, label="Fraud")

ax.set_xticks(x)
ax.set_xticklabels(means_0.index, rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(proj_root + "/plots/bar.png")
plt.close()

# Mean values seem to be significantly lower in most categories


# %%
def plot_class_diff_by_stats_measure(df1, df2, measure="mean", filename=None):
    fig, ax = plt.subplots(figsize=(14, 6))

    # Extract the mean values
    means_0 = df1.loc[measure, "V1":"V28"]
    means_1 = df2.loc[measure, "V1":"V28"]

    # Create x positions
    x = np.arange(len(means_0))
    width = 0.35

    # Plot grouped bars
    ax.bar(x - width / 2, means_0, width, label="Legitimate")
    ax.bar(x + width / 2, means_1, width, label="Fraud")

    ax.set_xticks(x)
    ax.set_xticklabels(means_0.index, rotation=45)
    plt.title(f"{measure}")
    ax.legend()
    plt.tight_layout()
    if filename == None:
        fig.savefig(proj_root + f"/plots/{measure}.png")
    else:
        fig.savefig(proj_root + f"/plots/{filename}.png")
    plt.close()


# %%
plot_class_diff_by_stats_measure(class_0_described, class_1_described, "mean")
plot_class_diff_by_stats_measure(class_0_described, class_1_described, "std")


# %% Undersampling
def compare_through_undersampling(df1, df2, measure="mean"):
    if len(df1) < len(df2):
        df2 = df2.sample(len(df1))
    elif len(df2) < len(df1):
        df1 = df1.sample(len(df2))

    df1_described = df1.describe()
    df2_described = df2.describe()

    # Plot the measure
    plot_class_diff_by_stats_measure(
        df1_described, df2_described, measure, filename=f"undersampled_{measure}.png"
    )


# %%
compare_through_undersampling(class_0_df, class_1_df, measure="std")

# %% Creating correlations
abs_corr = np.abs(raw_df.corr()["Class"])
sorted_abs_corr = abs_corr.sort_values(ascending=False).iloc[1:]
# sorted_abs_corr

fig, ax = plt.subplots(figsize=(14, 6))
plt.title("Feature Correalations to Class")
sorted_abs_corr.plot(kind="bar", ax=ax)
plt.savefig(proj_root + "/plots/correlations.png")
plt.close()
