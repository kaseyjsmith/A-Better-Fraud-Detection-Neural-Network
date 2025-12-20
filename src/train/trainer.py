# Torch / NN stuff
import torch
from torch.optim import Adam

# Math stuff
import numpy as np

# Evaluation and plotting
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# other
from dataclasses import dataclass
from typing import List
from pathlib import Path

try:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent.__str__()
except Exception as e:
    proj_root = "/home/ksmith/birds/neural_networks/fraud_detection"


@dataclass
class History:
    loss: List[float]
    precision: List[float]
    recall: List[float]
    f1: List[float]
    roc_auc: List[float]


class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        data_loader,
        test_data: tuple,
        epochs=50,
        device="cpu",
    ):
        self.model = model
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.train_loader = data_loader["train"]
        self.test_loader = data_loader["test"]
        # test_data is a tuple of X_test and y_test
        self.X_test, self.y_test = test_data
        self.device = device

        self.history = None

    def fit(
        self,
        learning_rate,
        epochs,
        batch_size=32,
        early_stopping_patience=None,
        save_best=True,
        save_dir=f"{proj_root}/models/saved_models",
    ):
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        epoch_loss = 0

        # Track best model for checkpointing
        best_roc_auc = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            # Train
            # set model to training mode
            epoch_loss = 0
            for batch_x, batch_y in self.train_loader:
                # Make predicitons
                predictions = self.model(batch_x)

                # Check those predictions with the loss function
                batch_loss = self.loss_fn(predictions, batch_y)

                # Backpropigation
                optimizer.zero_grad()  # Clear old gradients
                batch_loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

                epoch_loss += batch_loss.item()

            # Eval
            precision, recall, f1, roc_auc = self.evaluate(
                epoch, epoch_loss, threshold=0.9
            )
            # If it's the first pass, setup history
            if self.history == None:
                self.history = History(
                    loss=[epoch_loss],
                    precision=[precision],
                    recall=[recall],
                    f1=[f1],
                    roc_auc=[roc_auc],
                )
            # Else add to history
            else:
                self.history.loss.append(epoch_loss)
                self.history.precision.append(precision)
                self.history.recall.append(recall)
                self.history.f1.append(f1)
                self.history.roc_auc.append(roc_auc)

            # TODO(human): Implement model checkpointing
            # Check if current ROC-AUC is better than best_roc_auc
            # If yes:
            #   - Update best_roc_auc and best_epoch
            #   - Call self.save_checkpoint() with appropriate parameters
            #   - Print a message like "✓ New best model saved! ROC-AUC: {roc_auc:.4f}"
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_epoch = epoch
                best_metrics = {
                    "roc_auc": roc_auc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "loss": epoch_loss,
                }
                filepath = f"{save_dir}/best_model_lr{learning_rate}_roc_auc{roc_auc}.pt"
                self.save_checkpoint(
                    filepath=filepath,
                    epoch=best_epoch,
                    metrics=best_metrics,
                    learning_rate=learning_rate,
                )
                print(f"New best model saved! ROC-AUC: {roc_auc:.4f}")

        return self.history

    def evaluate(self, epoch, epoch_loss, threshold=0.5):
        # Set to eval mode
        self.model.eval()
        with torch.no_grad():
            test_predictions = self.model(self.X_test)
            test_predictions_prob = torch.sigmoid(test_predictions).numpy()
            test_predictions_binary = (
                test_predictions_prob > threshold
            ).astype(int)

            # Calculate metrics
            precision = precision_score(
                self.y_test, test_predictions_binary, zero_division=0
            )
            recall = recall_score(
                self.y_test, test_predictions_binary, zero_division=0
            )
            f1 = f1_score(self.y_test, test_predictions_binary, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, test_predictions_binary)

        print(
            f"Epoch {epoch}/{self.epochs} | Loss: {epoch_loss:.4f} | "
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}"
        )

        return precision, recall, f1, roc_auc

    def save_checkpoint(self, filepath, epoch, metrics, learning_rate):
        """
        Save model checkpoint with metadata.

        Args:
            filepath: Path to save the checkpoint (e.g., "models/best_model.pt")
            epoch: Current epoch number
            metrics: Dictionary of current metrics
            learning_rate: Learning rate used for training
        """
        from pathlib import Path

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save model state and metadata
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "learning_rate": learning_rate,
        }

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """
        Load model checkpoint.

        Args:
            filepath: Path to the checkpoint file

        Returns:
            Dictionary with epoch and metrics info
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "learning_rate": checkpoint.get("learning_rate", None),
        }
