#!/usr/bin/env python3
"""
Learning Rate Sweep Experiment

Tests multiple learning rates to find the optimal value for smooth convergence.
Tests: [0.0001, 0.0005, 0.001, 0.005]
"""

import joblib
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Setup paths
try:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent.__str__()
except Exception as e:
    proj_root = "/home/ksmith/birds/neural_networks/fraud_detection"

sys.path.insert(0, proj_root)

from src.models.nn.simple_nn import FraudDetectionNN
from src.train.trainer import Trainer

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================
# LEARNING_RATES = [0.0001, 0.0005, 0.001, 0.005]
LEARNING_RATES = [0.0001]
EPOCHS = 50
BATCH_SIZE = 32
THRESHOLD = 0.5

print("=" * 70)
print("LEARNING RATE SWEEP EXPERIMENT")
print("=" * 70)
print(f"\nTesting learning rates: {LEARNING_RATES}")
print(f"Epochs per experiment: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Evaluation threshold: {THRESHOLD}")
print("=" * 70)

# ============================================================
# DATA LOADING
# ============================================================
print("\n[1/4] Loading preprocessed data...")
X_train_scaled = joblib.load(f"{proj_root}/data/X_train_scaled.pkl")
X_test_scaled = joblib.load(f"{proj_root}/data/X_test_scaled.pkl")
y_train = joblib.load(f"{proj_root}/data/y_train.pkl")
y_test = joblib.load(f"{proj_root}/data/y_test.pkl")
print(f"      ✓ Loaded {len(X_train_scaled)} training samples")
print(f"      ✓ Loaded {len(X_test_scaled)} test samples")

# ============================================================
# SETUP LOSS FUNCTION & DATA LOADERS
# ============================================================
print("\n[2/4] Setting up training infrastructure...")

# Calculate class weights
fraud_count = (y_train == 1).sum()
total = len(y_train)
pos_weight = total / (2 * fraud_count)
loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
print(f"      ✓ Loss function with pos_weight={pos_weight:.2f}")

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"      ✓ DataLoader ready")

# ============================================================
# RUN EXPERIMENTS
# ============================================================
print("\n[3/4] Running learning rate experiments...")
print("-" * 70)

# Store results for each learning rate
results = {}

# TODO(human): Implement the learning rate sweep loop
model_runs = []
trainer_runs = []
for idx, lr in enumerate(LEARNING_RATES):
    #   1. Create a fresh FraudDetectionNN model (important: new model each time!)
    model_runs.append(FraudDetectionNN())
    current_model = model_runs[idx]

    #   2. Create a Trainer with the model, loss_fn, train_loader, and test data
    trainer_runs.append(
        Trainer(
            model=current_model,
            loss_fn=loss_fn,
            data_loader={"train": train_loader, "test": None},
            test_data=(X_test_tensor, y_test_tensor),
        )
    )
    trainer = trainer_runs[idx]

    #   3. Train using trainer.fit(learning_rate=lr, epochs=EPOCHS)
    # Train the model with the current lr
    print(
        f"\n{'=' * 70}\nExperiment {idx + 1}/4: Testing LR = {lr}\n{'=' * 70}"
    )
    trainer.fit(learning_rate=lr, epochs=EPOCHS)

    #   4. Store the history in results[lr] for later comparison.
    results[lr] = trainer.history


# For each learning rate in LEARNING_RATES:
#
# Hints:
#   - Use FraudDetectionNN() to create a new model
#   - Pass test_data as (X_test_tensor, y_test_tensor)
#   - Print progress between experiments to track which LR is running
#   - Store as: results[lr] = history

print("\n" + "-" * 70)
print("✓ All experiments complete!")

# ============================================================
# VISUALIZE RESULTS
# ============================================================
print("\n[4/4] Generating comparison plots...")

# Create a figure with subplots for each metric
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Learning Rate Comparison", fontsize=16, fontweight="bold")

metrics = ["loss", "precision", "recall", "f1", "roc_auc"]
metric_names = ["Loss", "Precision", "Recall", "F1-Score", "ROC-AUC"]

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes.flatten()[idx]

    for lr in LEARNING_RATES:
        if lr in results:
            history = results[lr]
            values = getattr(history, metric)
            ax.plot(values, label=f"LR={lr}", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(name)
    ax.set_title(f"{name} vs Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

# Hide the last subplot (we only have 5 metrics)
axes.flatten()[5].axis("off")

plt.tight_layout()

# Save the plot
output_path = f"{proj_root}/plots/lr_sweep_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"      ✓ Saved comparison plot to: plots/lr_sweep_comparison.png")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)

for lr in LEARNING_RATES:
    if lr in results:
        history = results[lr]
        print(f"\nLearning Rate: {lr}")
        print(f"  Final Loss:      {history.loss[-1]:.4f}")
        print(f"  Final Precision: {history.precision[-1]:.4f}")
        print(f"  Final Recall:    {history.recall[-1]:.4f}")
        print(f"  Final F1:        {history.f1[-1]:.4f}")
        print(f"  Final ROC-AUC:   {history.roc_auc[-1]:.4f}")
        print(
            f"  Avg Loss (last 10 epochs): {sum(history.loss[-10:]) / 10:.4f}"
        )

print("\n" + "=" * 70)
print(
    "Experiment complete! Check plots/lr_sweep_comparison.png for visual comparison."
)
print("=" * 70)
