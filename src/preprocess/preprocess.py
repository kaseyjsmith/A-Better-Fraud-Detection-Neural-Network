# %%
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import joblib

# %%
try:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent.__str__()
except Exception as e:
    proj_root = "/home/ksmith/birds/neural_networks/fraud_detection"

data_file = proj_root + "/data/creditcard.csv"
raw_df = pd.read_csv(data_file)

# %%
X = raw_df.iloc[:, :-1]
y = raw_df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# %%
scaler = StandardScaler()
scaler.fit(X_train)
# scaler.get_params()

# %%
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% Save the scaled data and scaler for later use
joblib.dump(scaler, proj_root + "src/models/scalers/scaler.pkl")
joblib.dump(X_train_scaled, proj_root + "/data/X_train_scaled.pkl")
joblib.dump(X_test_scaled, proj_root + "/data/X_test_scaled.pkl")
joblib.dump(y_train, proj_root + "/data/y_train.pkl")
joblib.dump(y_test, proj_root + "/data/y_test.pkl")
