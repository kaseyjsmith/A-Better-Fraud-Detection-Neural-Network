from pathlib import Path
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

# Setup paths
try:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent.parent.__str__()  # Go up to project root
except Exception as e:
    proj_root = "/home/ksmith/birds/neural_networks/fraud_detection"

import sys

sys.path.insert(0, proj_root)

from src.models.nn.lightning_fraud import FraudDetactionLightning

# Constants
EPOCHS = 50
BATCH_SIZE = 512  # Increased from 32 - more computation per batch
NUM_WORKERS = 8  # Parallel data loading across N CPU cores

# Load preprocessed data
X_train_scaled = joblib.load(proj_root + "/data/X_train_scaled.pkl")
X_test_scaled = joblib.load(proj_root + "/data/X_test_scaled.pkl")
y_train = joblib.load(proj_root + "/data/y_train.pkl")
y_test = joblib.load(proj_root + "/data/y_test.pkl")

# Calculate pos_weight for class imbalance (matches train.py:56-60)
fraud_count = (y_train == 1).sum()
total = len(y_train)
pos_weight = total / (2 * fraud_count)
print(
    f"Class imbalance - Total: {total}, Fraud: {fraud_count}, pos_weight: {pos_weight:.4f}"
)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

# Create datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create data loaders with parallel workers for faster data loading
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,  # Parallel data loading
    persistent_workers=True,  # Keep workers alive between epochs
    pin_memory=True,  # Faster CPU-to-GPU transfer (beneficial even on CPU)
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    pin_memory=True,
)

# Create the Lightning model with pos_weight for class imbalance
model = FraudDetactionLightning(pos_weight=pos_weight)

# Create Lightning Trainer
trainer = L.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",  # Automatically uses GPU if available, else CPU
    devices=1,
    log_every_n_steps=10,
)

# Train the model
print(f"\nStarting training for {EPOCHS} epochs...")
trainer.fit(model, train_loader)

# Test the model
print("\nRunning final test evaluation...")
results = trainer.test(model, test_loader)
