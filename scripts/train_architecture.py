import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


import argparse
import joblib
from pathlib import Path

try:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent.__str__()
except Exception as e:
    proj_root = "/home/ksmith/birds/neural_networks/fraud_detection"

import sys

sys.path.insert(0, proj_root)


from src.models.nn.architecture_factory import (
    create_architecture,
    list_architectures,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", nargs=1, help="Architecture argument")
    parser.add_argument("--epochs", nargs=1, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", nargs=1, type=int, help="Batch size")
    parser.add_argument(
        "--num_workers", nargs=1, type=int, help="Number of workers"
    )

    args = parser.parse_args()
    if args.arch:
        if args.epochs:
            epochs = args.epochs[0]
        else:
            epochs = 50
        if args.batch_size:
            BATCH_SIZE = args.batch_size[0]
        else:
            BATCH_SIZE = 512
        if args.num_workers:
            NUM_WORKERS = args.num_workers[0]
        else:
            NUM_WORKERS = 23  # CPU has 23 cores
        # load training data
        # Import the data and scaler
        X_train_scaled = joblib.load(proj_root + "/data/X_train_scaled.pkl")
        X_test_scaled = joblib.load(proj_root + "/data/X_test_scaled.pkl")

        y_train = joblib.load(proj_root + "/data/y_train.pkl")
        y_test = joblib.load(proj_root + "/data/y_test.pkl")

        scaler = joblib.load(proj_root + "/src/models/scalers/scaler.pkl")

        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # calculate pos_weight
        fraud_count = (y_train == 1).sum()
        legit_count = (y_train == 0).sum()
        total = len(y_train)
        pos_weight = total / (2 * fraud_count)
        print(
            f"Class imbalance - Total: {total}, Fraud: {fraud_count}, pos_weight: {pos_weight:.4f}"
        )

        model = create_architecture(args.arch[0], pos_weight=pos_weight)
        print(f"Model created: {type(model).__name__}")
        print(f"Model pos_weight in loss_fn: {model.loss_fn.pos_weight}")
        # Set up the lightning trainer
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
        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator="auto",  # Automatically uses GPU if available, else CPU
            devices=1,
            log_every_n_steps=10,
        )
        # Fit/train the model
        trainer.fit(model, train_loader)

        # Test the model
        print("\nRunning final test evaluation...")
        results = trainer.test(model, test_loader)
