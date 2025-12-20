#!/usr/bin/env python3
"""
Read and display metrics from Lightning training checkpoints.

This script scans the lightning_logs directory for all training versions,
extracts metrics from CSV logs and checkpoint files, and displays them
in a readable format for comparison.
"""

from pathlib import Path
import pandas as pd
import torch
from datetime import datetime

# Setup paths
try:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent
except Exception as e:
    proj_root = Path("/home/ksmith/birds/neural_networks/fraud_detection")

logs_dir = proj_root / "lightning_logs"


def read_version_metrics(version_dir):
    """
    Read metrics from a single version directory.

    Returns dict with version info and metrics, or None if no data found.
    """
    metrics_csv = version_dir / "metrics.csv"
    checkpoints_dir = version_dir / "checkpoints"

    result = {
        "version": version_dir.name,
        "metrics": None,
        "checkpoint": None,
    }

    # Read metrics CSV if it exists
    if metrics_csv.exists():
        try:
            df = pd.read_csv(metrics_csv)
            # Get the last row (final metrics)
            if len(df) > 0:
                result["metrics"] = df.iloc[-1].to_dict()
        except Exception as e:
            print(f"  ⚠ Error reading {metrics_csv}: {e}")

    # Find checkpoint file
    if checkpoints_dir.exists():
        ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
        if ckpt_files:
            # Get the most recent checkpoint
            ckpt_file = max(ckpt_files, key=lambda p: p.stat().st_mtime)
            result["checkpoint"] = {
                "file": ckpt_file.name,
                "size_mb": ckpt_file.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(ckpt_file.stat().st_mtime),
            }

            # Try to load checkpoint for additional info
            try:
                ckpt = torch.load(ckpt_file, map_location='cpu')
                result["checkpoint"]["epoch"] = ckpt.get("epoch", "?")
                result["checkpoint"]["global_step"] = ckpt.get("global_step", "?")
            except Exception as e:
                print(f"  ⚠ Could not load checkpoint {ckpt_file.name}: {e}")

    return result if (result["metrics"] or result["checkpoint"]) else None


def main():
    """Main function to read and display all checkpoint metrics."""

    print("=" * 80)
    print("Lightning Training Checkpoint Metrics")
    print("=" * 80)
    print()

    if not logs_dir.exists():
        print(f"❌ Lightning logs directory not found: {logs_dir}")
        return

    # Find all version directories
    version_dirs = sorted([d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("version_")])

    if not version_dirs:
        print(f"❌ No version directories found in {logs_dir}")
        return

    print(f"Found {len(version_dirs)} training versions\n")

    # Collect all results
    all_results = []
    for version_dir in version_dirs:
        result = read_version_metrics(version_dir)
        if result:
            all_results.append(result)

    # Display results
    if not all_results:
        print("❌ No metrics found in any version")
        return

    # Print summary table
    print("┌" + "─" * 78 + "┐")
    print("│" + " " * 28 + "TRAINING SUMMARY" + " " * 34 + "│")
    print("├" + "─" * 78 + "┤")
    print(f"│ {'Version':<12} {'Epoch':<8} {'F1':<10} {'Precision':<12} {'Recall':<10} {'ROC-AUC':<10} │")
    print("├" + "─" * 78 + "┤")

    for result in all_results:
        version = result["version"]

        if result["metrics"]:
            m = result["metrics"]
            epoch = int(m.get("epoch", 0))
            f1 = m.get("test_f1", 0)
            precision = m.get("test_precision", 0)
            recall = m.get("test_recall", 0)
            roc_auc = m.get("test_roc_auc", 0)

            print(f"│ {version:<12} {epoch:<8} {f1:<10.4f} {precision:<12.4f} {recall:<10.4f} {roc_auc:<10.4f} │")
        else:
            print(f"│ {version:<12} {'N/A':<8} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<10} │")

    print("└" + "─" * 78 + "┘")
    print()

    # Print detailed info for each version
    print("=" * 80)
    print("DETAILED METRICS BY VERSION")
    print("=" * 80)
    print()

    for result in all_results:
        version = result["version"]
        print(f"📊 {version.upper()}")
        print("─" * 80)

        # Checkpoint info
        if result["checkpoint"]:
            ckpt = result["checkpoint"]
            print(f"  Checkpoint: {ckpt['file']}")
            print(f"  Size: {ckpt['size_mb']:.2f} MB")
            print(f"  Modified: {ckpt['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
            print(f"  Global Step: {ckpt.get('global_step', 'N/A')}")
        else:
            print("  ⚠ No checkpoint found")

        print()

        # Metrics info
        if result["metrics"]:
            m = result["metrics"]
            print("  Test Metrics:")
            print(f"    Loss:      {m.get('test_loss', 'N/A'):.6f}")
            print(f"    Precision: {m.get('test_precision', 0):.4f}")
            print(f"    Recall:    {m.get('test_recall', 0):.4f}")
            print(f"    F1:        {m.get('test_f1', 0):.4f}")
            print(f"    ROC-AUC:   {m.get('test_roc_auc', 0):.4f}")
        else:
            print("  ⚠ No metrics found")

        print()

    # Find best performing model
    print("=" * 80)
    print("BEST PERFORMING MODELS")
    print("=" * 80)
    print()

    models_with_metrics = [r for r in all_results if r["metrics"]]

    if models_with_metrics:
        # Best F1 score
        best_f1 = max(models_with_metrics, key=lambda r: r["metrics"].get("test_f1", 0))
        print(f"🏆 Best F1 Score: {best_f1['version']}")
        print(f"   F1: {best_f1['metrics']['test_f1']:.4f}")

        # Best ROC-AUC
        best_roc = max(models_with_metrics, key=lambda r: r["metrics"].get("test_roc_auc", 0))
        print(f"\n🏆 Best ROC-AUC: {best_roc['version']}")
        print(f"   ROC-AUC: {best_roc['metrics']['test_roc_auc']:.4f}")

        # Best Recall (important for fraud detection!)
        best_recall = max(models_with_metrics, key=lambda r: r["metrics"].get("test_recall", 0))
        print(f"\n🏆 Best Recall (Fraud Detection): {best_recall['version']}")
        print(f"   Recall: {best_recall['metrics']['test_recall']:.4f}")
        print(f"   (Catches {best_recall['metrics']['test_recall']*100:.1f}% of fraud cases)")


if __name__ == "__main__":
    main()
