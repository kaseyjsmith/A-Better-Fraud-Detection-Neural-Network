from typing import List
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

try:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent.__str__()
except Exception as e:
    proj_root = "/home/ksmith/birds/neural_networks/fraud_detection"


class Recorder:
    def __init__(self, run_id):
        self.run_id = run_id

    def set_loss_history(self, loss_history):
        self.loss_history = loss_history

    def set_precision_history(self, precision_history):
        self.precision_history = precision_history

    def set_recall_history(self, recall_history):
        self.recall_history = recall_history

    def set_f1_history(self, f1_history):
        self.f1_history = f1_history

    def set_roc_auc_history(self, roc_auc_history):
        self.roc_auc_history = roc_auc_history

    def get_metrics(self):
        metrics = []
        if hasattr(self, "loss_history"):
            metrics.append(self.loss_history)
        if hasattr(self, "precision_history"):
            metrics.append(self.precision_history)
        if hasattr(self, "recall_history"):
            metrics.append(self.recall_history)
        if hasattr(self, "f1_history"):
            metrics.append(self.f1_history)
        if hasattr(self, "roc_auc_history"):
            metrics.append(self.roc_auc_history)
        return metrics

    def set_all_metrics(self, **metrics):
        if "loss_history" in metrics:
            self.set_loss_history(metrics["loss_history"])
        if "precision_history" in metrics:
            self.set_precision_history(metrics["precision_history"])
        if "recall_history" in metrics:
            self.set_recall_history(metrics["recall_history"])
        if "f1_history" in metrics:
            self.set_f1_history(metrics["f1_history"])
        if "roc_auc_history" in metrics:
            self.set_roc_auc_history(metrics["roc_auc_history"])

    def plot_metrics(self):
        """
        Plots the metrics in Recorder object. Must be run after setting metric histories in Recorder object.

        Args:
          None
        """
        fig, ax = plt.subplots()

        # Loss is unbounded in y
        ax.plot(self.loss_history, label="Loss", color="red")
        ax.set_ylabel("Loss", color="red")
        ax.tick_params(axis="y", labelcolor="red")

        # break out these metrics onto separate y-axis as they all range from 0 to 1
        ax2 = ax.twinx()
        ax2.plot(self.precision_history, label="Precision", color="blue")
        ax2.plot(self.recall_history, label="Recall", color="green")
        ax2.plot(self.f1_history, label="F1", color="orange")
        ax2.plot(self.roc_auc_history, label="ROC-AUC", color="purple")
        ax2.set_ylabel("Metric Value", color="black")
        ax2.set_ylim((0, 1))

        ax.set_xlabel("Epoch")

        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.title(f"{self.run_id} Performance Metrics")
        fig.tight_layout()
        fig.savefig(proj_root + f"/plots/{self.run_id}_metrics.png")

    def save_metrics(self):
        # TODO: check if all of the metrics are none

        joblib.dump(
            [
                self.loss_history,
                self.precision_history,
                self.recall_history,
                self.f1_history,
                self.roc_auc_history,
            ],
            f"{self.run_id}_metrics.pkl",
        )
