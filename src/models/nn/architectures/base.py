import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from uuid import uuid4

import lightning as L

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class BaseFraudNN(L.LightningModule):
    def __init__(self, pos_weight=None, lr=0.008):
        super().__init__()
        self.pos_weight = pos_weight
        self.lr = lr

        # Set up loss function with class weighting for imbalanced data
        if pos_weight is not None:
            self.loss_fn = BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
            )
        else:
            self.loss_fn = BCEWithLogitsLoss()

        self.run_id = uuid4()

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        # Learning rate scaled for batch size
        # Base lr=0.001 for batch_size=32, scale linearly: 512/32 = 16x → lr=0.016
        # Using conservative 0.008 (half of linear scaling) to avoid instability
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Unpack batch, run forward pass, compute loss, return loss
        batch_x, batch_y = batch
        predictions = self.forward(batch_x)
        loss = self.loss_fn(predictions, batch_y)
        # Lightning automatically calls backwards() to find gradients when loss is returned
        # Then updates weights (optimizer.step()) and then clears gradients for the next pass (optimizer.zero_grad())
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack the batch into features and labels
        batch_x, batch_y = batch

        # Forward pass - get model predictions (logits)
        predictions = self(batch_x)

        # Calculate loss for this batch
        loss = self.loss_fn(predictions, batch_y)

        # Convert logits to probabilities using sigmoid activation
        # BCEWithLogitsLoss expects logits, but metrics need probabilities
        probs = torch.sigmoid(predictions)

        # Convert probabilities to binary predictions using 0.5 threshold
        binary_preds = (probs > 0.5).float()

        # Log the validation loss to Lightning's progress bar
        self.log("val_loss", loss, prog_bar=True)

        # Collect outputs for epoch-level metric calculation
        # Store predictions and targets from this batch
        output = {"preds": binary_preds, "probs": probs, "targets": batch_y}
        self.validation_step_outputs.append(output)

        return loss

    def test_step(self, batch, batch_idx):
        # Unpack the batch into features and labels
        batch_x, batch_y = batch

        # Forward pass - get model predictions (logits)
        predictions = self(batch_x)

        # Calculate loss for this batch
        loss = self.loss_fn(predictions, batch_y)

        # Convert logits to probabilities using sigmoid activation
        probs = torch.sigmoid(predictions)

        # Convert probabilities to binary predictions using 0.5 threshold
        binary_preds = (probs > 0.9).float()

        # Log the test loss to Lightning's progress bar
        self.log("test_loss", loss, prog_bar=True)

        # Collect outputs for epoch-level metric calculation
        output = {"preds": binary_preds, "probs": probs, "targets": batch_y}
        self.test_step_outputs.append(output)

        return loss

    def on_validation_epoch_end(self):
        # This method is called after all validation_step calls
        # Aggregate all batch outputs into complete dataset predictions

        # Concatenate all batch predictions and targets into single tensors
        all_preds = torch.cat(
            [x["preds"] for x in self.validation_step_outputs]
        )
        all_probs = torch.cat(
            [x["probs"] for x in self.validation_step_outputs]
        )
        all_targets = torch.cat(
            [x["targets"] for x in self.validation_step_outputs]
        )

        # Convert to numpy for sklearn metrics (move to CPU if on GPU)
        preds_np = all_preds.cpu().numpy()
        probs_np = all_probs.cpu().numpy()
        targets_np = all_targets.cpu().numpy()

        # Compute metrics on the full validation set (matches train.py:112-117)
        precision = precision_score(targets_np, preds_np, zero_division=0)
        recall = recall_score(targets_np, preds_np, zero_division=0)
        f1 = f1_score(targets_np, preds_np, zero_division=0)
        roc_auc = roc_auc_score(targets_np, probs_np)

        # Log metrics to Lightning (will appear in progress bar and logs)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_roc_auc", roc_auc, prog_bar=True)

        # Clear the outputs list for the next epoch
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        # This method is called after all test_step calls
        # Aggregate all batch outputs into complete dataset predictions

        # Concatenate all batch predictions and targets
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        all_probs = torch.cat([x["probs"] for x in self.test_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.test_step_outputs])

        # Convert to numpy for sklearn metrics
        preds_np = all_preds.cpu().numpy()
        probs_np = all_probs.cpu().numpy()
        targets_np = all_targets.cpu().numpy()

        # Compute metrics on the full test set
        precision = precision_score(targets_np, preds_np, zero_division=0)
        recall = recall_score(targets_np, preds_np, zero_division=0)
        f1 = f1_score(targets_np, preds_np, zero_division=0)
        roc_auc = roc_auc_score(targets_np, probs_np)

        # Log metrics to Lightning
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_recall", recall, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_roc_auc", roc_auc, prog_bar=True)

        # Clear the outputs list
        self.test_step_outputs.clear()
