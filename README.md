# Credit Card Fraud Detection with Neural Networks

I wanted to make a project that encompassed 3 things:

- An exploration of different neural networks and their tradeoffs
- Perform this exploration on a dataset that was highly class imbalanced
- Do this in a production-level project, not just another Jupyter notebook

This project demonstrates a systematic approach to neural network architecture experimentation for fraud detection. Using the Credit Card fraud dataset, I explore how different network architectures perform on highly imbalanced data (~0.2% fraud) with minimal feature engineering.

## Project Overview

**Goal**: Compare multiple neural network architectures and parameter tuning to understand architectural tradeoffs for fraud detection

**Approach**: Incremental experimentation. Build one architecture at a time, train it, analyze results, and learn from each iteration

**Current Baseline**: Simple NN with layer 30→64→32→16→8→1 architecture achieving F1≈0.80, Precision≈82%, Recall≈78%

---

## Quick Start

### 1. Setup Environment

```bash
# Activate your virtual environment
source ~/.venvs/data_stuffs/bin/activate  # or your venv path

# Install dependencies
pip install torch lightning scikit-learn pandas joblib matplotlib seaborn
```

### 2. Prepare Data

```bash
# (Optional) Run exploratory data analysis
python src/explore.py
```

```bash
# Preprocess the raw dataset (one-time setup)
python src/preprocess.py

```

### 3. Train an Architecture

```bash
# Train the baseline architecture for 50 epochs
# All architectures available in src/models/nn/architecture_factory.py
python -m scripts.train_architecture --arch baseline --epochs 50

# Results will be logged to lightning_logs/ and experiments/archtecture_runs.txt
```

### 4. Add a New Architecture

See [Adding New Architectures](#adding-new-architectures) below.

---

## Project Structure

```
fraud_detection/
├── data/                              # Dataset and preprocessed files
│   ├── creditcard.csv                 # Raw dataset (~285K transactions, 30 features)
│   ├── X_train_scaled.pkl             # Preprocessed training features
│   ├── X_test_scaled.pkl              # Preprocessed test features
│   ├── y_train.pkl                    # Training labels
│   └── y_test.pkl                     # Test labels
│
├── src/
│   ├── explore.py                     # Exploratory data analysis
│   ├── preprocess.py                  # Data loading, scaling, train/test split
│   │
│   └── models/nn/
│       ├── architecture_factory.py    # Factory for creating architectures
│       │
│       └── architectures/
│           ├── base.py                # Base class with shared training logic
│           ├── baseline.py            # Baseline: 30→64→32→16→8→1
│           ├── wide_network.py        # (Coming) Wide: 30→128→128→64→32→1
│           ├── deep_network.py        # (Coming) Deep: 8 layers
│           ├── resnet_fraud.py        # (Coming) ResNet-style with skip connections
│           ├── batchnorm_network.py   # (Coming) With BatchNorm
│           └── layernorm_network.py   # (Coming) With LayerNorm
│
├── scripts/
│   ├── train_architecture.py          # CLI trainer for any architecture
│   └── read_checkpoint_metrics.py     # Utility to read Lightning checkpoints
│
├── plots/                             # EDA visualizations
├── lightning_logs/                    # Training logs and checkpoints
├── ARCHITECTURE_PLAN.md               # Experimentation roadmap and results
└── README.md                          # This file
```

---

## Architecture Experimentation Framework

### Design Philosophy

The project uses a **Template Method Pattern** to enable rapid architecture experimentation:

- **`base.py`**: Defines shared training logic (optimizer, loss, metrics, Lightning hooks)
- **Architecture files**: Implement only `__init__` (layer definitions) and `forward` (forward pass)
- **Factory pattern**: Centralized architecture creation via string names
- **CLI trainer**: Unified interface to train any architecture

### Current Baseline Performance

**Architecture**: 30→64→32→16→8→1 (bottleneck design)
**Hyperparameters**: batch_size=512, lr=0.008, epochs=50, pos_weight≈289
**Results**:

- F1-Score: 0.79-0.80
- Precision: 78-82%
- Recall: 78-79%
- ROC-AUC: 0.89-0.90

### Class Imbalance Strategy

The dataset is highly imbalanced (~0.2% fraud). I handle this through:

1. **Weighted Loss**: `BCEWithLogitsLoss` with `pos_weight` parameter (makes fraud misclassification 285× more costly)
2. **Stratified Split**: Maintains class ratio in train/test sets
3. **Dropout Regularization**: 20% dropout prevents overfitting to majority class
   a. This could use optimizing. 20% is an educated first guess.
4. **Appropriate Metrics**: F1, Precision, Recall, ROC-AUC (not accuracy)

---

## Workflow & Commands

### Data Preprocessing

```bash
python src/preprocess.py
```

**What it does**:

- Loads `data/creditcard.csv`
- Stratified train/test split (maintains class imbalance ratio)
- StandardScaler normalization (fitted on training data only to prevent data leakage)
- Saves: `X_train_scaled.pkl`, `X_test_scaled.pkl`, `y_train.pkl`, `y_test.pkl`, scaler

**NOTE:** Run this before training.

### Exploratory Data Analysis

```bash
python src/explore.py
```

**What it does**:

- Generates statistical summaries
- Creates visualizations (correlation heatmaps, distributions by class)
- Outputs plots to `plots/` directory

This file was used while I was exploring the dataset. As I wanted to avoid using a Jupyter Notebook as described above, this is split into cells and was run in a REPL.

### Training an Architecture

```bash
# Basic usage
python -m scripts.train_architecture --arch baseline --epochs 50

# With custom hyperparameters
python -m scripts.train_architecture --arch baseline --epochs 100 --batch_size 256 --num_workers 8

# Available arguments:
#   --arch         Architecture name (baseline, wide, deep, etc.)
#   --epochs       Number of training epochs (default: 50)
#   --batch_size   Batch size (default: 512)
#   --num_workers  Number of data loading workers (default: 23)
```

**What it does**:

- Loads preprocessed data
- Calculates `pos_weight` from training labels
- Creates model via factory
- Trains with PyTorch Lightning
- Runs final test evaluation
- Saves logs and checkpoints to `lightning_logs/version_N/`

### Viewing Training Results

```bash
# View metrics from latest run
tail -5 lightning_logs/version_*/metrics.csv | grep test

# Read checkpoint details
python scripts/read_checkpoint_metrics.py
```

```bash
# Formatted easiest to view
cat experiements/architecture_runs.txt
```

---

## Adding New Architectures

### Step 1: Create Architecture File

Create `src/models/nn/architectures/your_architecture.py`:

```python
"""
Your Architecture Description

Architecture Details:
- Hypothesis: Why this architecture might help
- Tradeoffs: What you're giving up / gaining
"""

from .base import BaseFraudNN
import torch.nn as nn
import torch.nn.functional as F


class YourArchitectureNN(BaseFraudNN):
    def __init__(self, pos_weight=None, **kwargs):
        # Initialize base class (sets up loss, metrics, etc.)
        super().__init__(pos_weight=pos_weight)

        # Define your layers
        self.layer1 = nn.Linear(30, 128)  # Example
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

        # Optional: dropout, normalization, etc.
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Define forward pass
        x = F.relu(self.layer1(x))
        x = self.dropout(x)

        x = F.relu(self.layer2(x))
        x = self.dropout(x)

        return self.output(x)  # Return logits (no activation)
```

### Step 2: Register in Factory

Edit `src/models/nn/architecture_factory.py`:

```python
from .architectures.baseline import BaselineFraudNN
from .architectures.your_architecture import YourArchitectureNN  # Add import

ARCHITECTURES = {
    "baseline": BaselineFraudNN,
    "your_arch": YourArchitectureNN,  # Add to registry
}
```

### Step 3: Train and Analyze

```bash
# Train your new architecture
python -m scripts.train_architecture --arch your_arch --epochs 50

# Compare results against baseline
# Document findings in ARCHITECTURE_PLAN.md
```

---

## Key Technical Details

### Data Characteristics

- **Size**: ~285,000 transactions
- **Features**: 30 (28 PCA-transformed V1-V28 + Time + Amount)
- **Imbalance**: ~0.2% fraud (positive class), 99.8% legitimate (negative class)
- **Class weighting**: `pos_weight = total / (2 * fraud_count)` ≈ 289

### Model Architecture Pattern

**Baseline: Bottleneck Design**

- Input: 30 features
- Hidden layers: 64 → 32 → 16 → 8 neurons (decreasing width)
- Activation: ReLU after each hidden layer
- Dropout: 20% after each activation
- Output: 1 neuron with logits (no activation - used with BCEWithLogitsLoss)

**Design Rationale**: Bottleneck architecture compresses features progressively; dropout prevents overfitting to majority class

### Training Setup

- **Loss**: `BCEWithLogitsLoss` with `pos_weight` for class imbalance
- **Optimizer**: Adam (lr=0.008, scaled for batch size 512)
- **Batch size**: 512 (increased from 32 for faster training)
- **Epochs**: 50
- **Metrics tracked**: Precision, Recall, F1-score, ROC-AUC
- **Framework**: PyTorch Lightning (handles training loop, device management, logging)

### Critical Implementation Notes

**Data Leakage Prevention**:

- `StandardScaler` is fitted on training data only, then applied to test data
- This prevents test set statistics from influencing the scaler

**Precision-Recall Tradeoff**:

- The 0.9 decision threshold can be tuned based on business requirements
  - For this dataset, a decision threashold of ~0.9 was found to be suitable
- I've taken the Product Management approach (as I am a Product Manager professionally) to prioritize precision over recall.
  - I would rather miss some and not nuissance customers rather than create unnecessary false alarms. In the end, some prediction capability and catching fraud is better than nothing. Whereas false alarms get people's suspicion up and is an unnecessarily scary experience in a false alarm.

**Dtype Handling**:

- Always use `torch.tensor([value], dtype=torch.float32)` for scalar tensors
- Mixing float64 (NumPy/Pandas default) with float32 (PyTorch default) causes training instability

---

## Experimentation Roadmap

See `ARCHITECTURE_PLAN.md` for the full experimentation plan. Summary:

**Completed**:

- [x] Foundation infrastructure (base class, factory, CLI)
- [x] Baseline architecture (F1≈0.80)

**Planned Architectures**:

1. [x] **Wide Network**: 30→128→128→64→32→1 (more capacity)
2. [x] **Deep Network**: 30→64→64→32→32→16→16→8→1 (deeper hierarchies)
3. [x] **ResNet-Style**: Skip connections for better gradient flow
4. [x] **BatchNorm**: Normalization for training stability
5. [ ] **LayerNorm**: Alternative normalization for imbalanced data

**Success Criteria**:

- Each architecture trains without errors
- At least one improves upon baseline F1≈0.80
- Clear understanding of architectural tradeoffs

---

## Learning Insights

### Imbalanced Classification

- Standard accuracy metric is misleading
  - A model could predict all as not fraud and be 99.8% accurate
- Always use F1, Precision/Recall, ROC-AUC for imbalanced problems
  - Allows for viewing important dimensions of overall "accuracy" and to perform the precision/recall tradeoff
- Weighted loss functions are crucial

### Template Method Pattern

- Separates "what to do" (training logic in base) from "how to compute" (architecture in children)
- Enables rapid experimentation without code duplication
- New architectures are ~15 lines instead of 200+ and avoids unnecessary duplicative code
  - Practicing good inheritance patterns

### PyTorch Lightning (and manual training pattern)

- Abstracts away boilerplate (training loops, device management, logging)
  - A manual trainer class is defined in `src/train/trainer.py` as I wanted to ensure I understood what Lightning is doing under the hood
  - Lightning is an emerging time saver and I wanted to learn the framework
- Provides consistent interface for experiments
- Built-in checkpointing and metric tracking

### Data Preprocessing

- Fit scalers/transformers on training data only
- Validate that preprocessing doesn't leak information from test set
- Stratified splits maintain class balance across splits

---

## Findings

See `architecture_dashboard.html`

## Citation

Dataset: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Research for unbalanced classification:

- Original Paper: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
