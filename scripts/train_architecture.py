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

import sys, os
from datetime import datetime
import ast

sys.path.insert(0, proj_root)


from src.models.nn.architecture_factory import (
    create_architecture,
    list_architectures,
)


def training_setup(**config):
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

    model = create_architecture(config["architecture"], pos_weight=pos_weight)
    print(f"Model created: {type(model).__name__}")
    print(f"Model pos_weight in loss_fn: {model.loss_fn.pos_weight}")
    # Set up the lightning trainer
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],  # Parallel data loading
        persistent_workers=True,  # Keep workers alive between epochs
        pin_memory=True,  # Faster CPU-to-GPU transfer (beneficial even on CPU)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        persistent_workers=True,
        pin_memory=True,
    )
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",  # Automatically uses GPU if available, else CPU
        devices=1,
        log_every_n_steps=10,
    )

    return model, trainer, train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", nargs=1, help="Architecture argument")
    parser.add_argument("--epochs", nargs=1, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", nargs=1, type=int, help="Batch size")
    parser.add_argument(
        "--num_workers", nargs=1, type=int, help="Number of workers"
    )
    parser.add_argument(
        "--learning_rate",
        nargs=1,
        type=float,
        help="OPTIONAL: learning rate for the optimizer of the base class",
    )

    args = parser.parse_args()
    config = {}
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

        config["architecture"] = args.arch[0]
        config["epochs"] = epochs
        config["batch_size"] = BATCH_SIZE
        config["num_workers"] = NUM_WORKERS
        if args.learning_rate:
            config["learning_rate"] = args.learning_rate[0]

        model, trainer, train_loader, test_loader = training_setup(**config)

        # Fit/train the model
        # TODO: setup a timer to train the model and output in arch_runs file
        # TODO: record average iterations per second
        trainer.fit(model, train_loader)

        # Test the model
        print("\nRunning final test evaluation...")
        results = trainer.test(model, test_loader)

        # write results to a testing file
        arch_runs = "experiments/architecture_runs.txt"
        # create the file if it doesn't exist
        if not os.path.exists(arch_runs):
            file = open(arch_runs, "w")
            file.close()
        # append the results to the file
        with open(arch_runs, "a") as file:
            file.write(f"{'=' * 80}\n")
            # Write model info and parameters
            file.write(f"Model: {type(model).__name__}\n")
            file.write(f"Model params:\n")
            file.write(f"    {model.parameters}\n")

            # Write the config used for the training run
            file.write(f"Training config:\n")
            for key in config:
                file.write(f"    {key}: {config[key]}\n")

            # Write when the run happened
            file.write(f"Run datetime: {datetime.now()}\n")

            # Write the results
            results_dict = ast.literal_eval(str(results[0]))
            for key in results_dict:
                file.write(f"    {key}: {results_dict[key]}\n")
