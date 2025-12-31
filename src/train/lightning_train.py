from pathlib import Path
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

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
ARCHITECTURE = "baseline"  # Options: "baseline", "wide", "deep", "resnet", etc.
EPOCHS = 50
BATCH_SIZE = 512  # Increased from 32 - more computation per batch
LEARNING_RATE = 0.008
NUM_WORKERS = 8  # Parallel data loading across N CPU cores


def generate_run_id(architecture, epochs, lr, batch_size):
    """
    Generate a descriptive run ID based on hyperparameters.

    Example: "baseline_e50_lr0.008_b512"
    """
    return f"{architecture}_e{epochs}_lr{lr}_b{batch_size}"


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

# Generate descriptive run ID
run_id = generate_run_id(ARCHITECTURE, EPOCHS, LEARNING_RATE, BATCH_SIZE)
print(f"Run ID: {run_id}")

# Create the Lightning model with pos_weight for class imbalance
model = FraudDetactionLightning(
    pos_weight=pos_weight, lr=LEARNING_RATE, run_id=run_id
)

# Create CSVLogger with descriptive name (prevents version_0, version_1, etc.)
logger = CSVLogger(
    save_dir=f"{proj_root}/lightning_logs",
    name=run_id,  # This becomes the folder name
)

# Create Lightning Trainer
trainer = L.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",  # Automatically uses GPU if available, else CPU
    devices=1,
    log_every_n_steps=10,
    logger=logger,  # Use custom logger
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,  # Wait 10 epochs before stopping
            verbose=True,  # Print when early stopping is triggered
        )
    ],
)

# Train the model
print(f"\nStarting training for {EPOCHS} epochs...")
trainer.fit(model, train_loader, test_loader)  # Pass test_loader as validation

# Test the model (this will use the best checkpoint if early stopping triggered)
print("\nRunning final test evaluation...")
results = trainer.test(model, test_loader)
