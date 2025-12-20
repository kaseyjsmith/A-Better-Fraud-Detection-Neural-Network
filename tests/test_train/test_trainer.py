#!/usr/bin/env python3
"""
Simple test script for the Trainer class.
Tests basic training loop with a small number of epochs.
"""

import joblib
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Setup paths
try:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.__str__()
except Exception as e:
    proj_root = "/home/ksmith/birds/neural_networks/fraud_detection"

import sys

sys.path.insert(0, proj_root)

from src.models.nn.simple_nn import FraudDetectionNN
from src.models.nn.lightning_fraud import FraudDetactionLightning
from src.train.trainer import Trainer

# Constants for testing
EPOCHS = 5  # Just a few epochs for testing
BATCH_SIZE = 32
LEARNING_RATE = 0.001

print("=" * 60)
print("Testing Trainer Class")
print("=" * 60)

# Load preprocessed data
print("\n1. Loading preprocessed data...")
X_train_scaled = joblib.load(f"{proj_root}/data/X_train_scaled.pkl")
X_test_scaled = joblib.load(f"{proj_root}/data/X_test_scaled.pkl")
y_train = joblib.load(f"{proj_root}/data/y_train.pkl")
y_test = joblib.load(f"{proj_root}/data/y_test.pkl")
print(f"   ✓ Loaded {len(X_train_scaled)} training samples")
print(f"   ✓ Loaded {len(X_test_scaled)} test samples")

# Create model
print("\n2. Initializing model...")
# model = FraudDetectionNN()
model = FraudDetactionLightning()
print(f"   ✓ Created FraudDetectionNN")

# Setup loss function with class weighting
print("\n3. Setting up loss function...")
fraud_count = (y_train == 1).sum()
total = len(y_train)
pos_weight = total / (2 * fraud_count)
loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
print(f"   ✓ BCEWithLogitsLoss with pos_weight={pos_weight:.2f}")

# Convert to PyTorch tensors
print("\n4. Converting to tensors...")
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
print(f"   ✓ Created PyTorch tensors")

# Create DataLoader
print("\n5. Creating DataLoader...")
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"   ✓ DataLoader with batch_size={BATCH_SIZE}")

# Create Trainer
print("\n6. Initializing Trainer...")
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    data_loader={"train": train_loader, "test": None},
    test_data=(X_test_tensor, y_test_tensor),
    epochs=EPOCHS,
    device="cpu",
)
print(f"   ✓ Trainer initialized for {EPOCHS} epochs")

# Train the model
print("\n7. Starting training...")
print("-" * 60)
try:
    history = trainer.fit(learning_rate=LEARNING_RATE, epochs=EPOCHS)
    print("-" * 60)
    print("\n✓ Training completed successfully!")
    print(f"\nFinal metrics:")
    print(f"  Loss: {history.loss[-1]:.4f}")
    print(f"  Precision: {history.precision[-1]:.4f}")
    print(f"  Recall: {history.recall[-1]:.4f}")
    print(f"  F1: {history.f1[-1]:.4f}")
    print(f"  ROC-AUC: {history.roc_auc[-1]:.4f}")
except Exception as e:
    print("-" * 60)
    print(f"\n✗ Training failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback

    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
