# %%
###
# IMPORTS
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, TensorDataset
from torch import tensor

try:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent.__str__()
except Exception as e:
    proj_root = "/home/ksmith/birds/neural_networks/fraud_detection"

import sys

sys.path.insert(0, proj_root)
from src.models.nn.simple_nn import FraudDetectionNN
from src.record.record import Recorder
###

###
# CONSTANTS
EPOCHS = 50
BATCH_SIZE = 32
###

# Import the data and scaler
X_train_scaled = joblib.load(proj_root + "/data/X_train_scaled.pkl")
X_test_scaled = joblib.load(proj_root + "/data/X_test_scaled.pkl")

y_train = joblib.load(proj_root + "/data/y_train.pkl")
y_test = joblib.load(proj_root + "/data/y_test.pkl")

scaler = joblib.load(proj_root + "/src/models/scalers/scaler.pkl")
# %%
# Create an instance of FraudDetectionNN
nn = FraudDetectionNN()  # All defaults for now

# Set up the optimizer
optimizer = Adam(nn.parameters(), lr=0.001)

# Set up loss BCEWithLogitsLoss with class weighting
fraud_count = (y_train == 1).sum()
legit_count = (y_train == 0).sum()
total = len(y_train)
pos_weight = total / (2 * fraud_count)
loss = BCEWithLogitsLoss(pos_weight=tensor([pos_weight]))


# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(
    -1, 1
)  # Reshape to [N, 1]

X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
loss_history = []
precision_history = []
recall_history = []
f1_history = []
roc_auc_history = []
recorder = Recorder(nn.run_id)

for epoch in range(EPOCHS):
    # Training phase

    nn.train()  # Set model to training mode (enables dropout)
    epoch_loss = 0

    for batch_x, batch_y in train_loader:
        # Forward pass
        predictions = nn(batch_x)

        # Calculate loss
        batch_loss = loss(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()  # Clear old gradients
        batch_loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        epoch_loss += batch_loss.item()

    # Evaluation phase
    nn.eval()  # Set model to evaluation mode (disables dropout)
    with torch.no_grad():  # Don't compute gradients during evaluation
        test_predictions = nn(X_test_tensor)
        test_predictions_prob = torch.sigmoid(test_predictions).numpy()
        test_predictions_binary = (test_predictions_prob > 0.5).astype(int)

        # Calculate metrics
        precision = precision_score(
            y_test, test_predictions_binary, zero_division=0
        )
        recall = recall_score(y_test, test_predictions_binary, zero_division=0)
        f1 = f1_score(y_test, test_predictions_binary, zero_division=0)
        roc_auc = roc_auc_score(y_test, test_predictions_prob)

    # Print progress
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss:.4f} | "
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}"
    )
    loss_history.append(epoch_loss)
    precision_history.append(precision)
    recall_history.append(recall)
    f1_history.append(f1)
    roc_auc_history.append(roc_auc)

# Create a plot of each metric over time
recorder.set_all_metrics(
    loss_history=loss_history,
    precision_history=precision_history,
    recall_history=recall_history,
    f1_history=f1_history,
    roc_auc_history=roc_auc_history,
)

# Plot the metrics
recorder.plot_metrics()


# %%
def test_threshold_values(step=0.01):
    precision_plot = []
    recall_plot = []
    with torch.no_grad():
        fig, ax = plt.subplots()
        for threshold in np.arange(0.01, 1, step):
            test_predictions = nn(X_test_tensor)
            test_predictions_prob = torch.sigmoid(test_predictions).numpy()
            test_predictions_binary = (
                test_predictions_prob > threshold
            ).astype(int)
            # Calculate metrics
            precision = precision_score(
                y_test, test_predictions_binary, zero_division=0
            )
            recall = recall_score(
                y_test, test_predictions_binary, zero_division=0
            )
            precision_plot.append(precision)
            recall_plot.append(recall)
            # f1 = f1_score(y_test, test_predictions_binary, zero_division=0)
            # roc_auc = roc_auc_score(y_test, test_predictions_prob)

            # Print progress
            # print(
            #     f"Threshold: {threshold:.1f} "
            #     f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
            #     f"F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}"
            # )
    ax.plot(precision_plot, recall_plot)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    plt.title(f"Thresholds for Sigmoid ranging from 0.01 to 1 in 0.01 steps")
    fig.savefig(proj_root + f"/plots/{nn.run_id}_recall_vs_precision.png")
    return precision_plot, recall_plot


precision_over_thresh, recall_over_thresh = test_threshold_values(step=0.01)

# %%

# Evaluat3
# - During training: Loss (to ensure it's decreasing)
# - On test set each epoch: Precision, Recall, F1, ROC-AUC
# - Final evaluation: Confusion matrix, detailed PR breakdown
