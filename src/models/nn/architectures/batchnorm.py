from .base import BaseFraudNN
from torch.nn import Dropout, Linear, BatchNorm1d
import torch.nn.functional as F


class BatchNormFraudNN(BaseFraudNN):
    """
    Deep network with Batch Normalization for training stability.

    BatchNorm normalizes activations between layers, which:
    - Prevents exploding/vanishing gradients (the problem with simple deep networks)
    - Allows higher learning rates
    - Widens the viable training window (can train for 50+ epochs, solves the problem with ResNet)
    - Acts as additional regularization

    Same depth as DeepFraudNN (11 layers) to enable direct comparison.
    """

    def __init__(
        self,
        input=30,
        h1=64,
        h2=64,
        h3=64,
        h4=64,
        h5=64,
        h6=32,
        h7=32,
        h8=32,
        h9=16,
        h10=16,
        out=1,
        dropout_rate=0.2,
        pos_weight=None,
    ):
        super().__init__(pos_weight)

        # Layers
        self.input = Linear(input, h1)
        self.h1 = Linear(h1, h2)
        self.h2 = Linear(h2, h3)
        self.h3 = Linear(h3, h4)
        self.h4 = Linear(h4, h5)
        self.h5 = Linear(h5, h6)
        self.h6 = Linear(h6, h7)
        self.h7 = Linear(h7, h8)
        self.h8 = Linear(h8, h9)
        self.h9 = Linear(h9, h10)
        self.h10 = Linear(h10, out)

        # Batch normalization layers (one after each hidden layer)
        # BatchNorm1d normalizes across the batch dimension
        self.bn_input = BatchNorm1d(h1)
        self.bn1 = BatchNorm1d(h2)
        self.bn2 = BatchNorm1d(h3)
        self.bn3 = BatchNorm1d(h4)
        self.bn4 = BatchNorm1d(h5)
        self.bn5 = BatchNorm1d(h6)
        self.bn6 = BatchNorm1d(h7)
        self.bn7 = BatchNorm1d(h8)
        self.bn8 = BatchNorm1d(h9)
        self.bn9 = BatchNorm1d(h10)

        self.dropout_rate = dropout_rate
        self.dropout = Dropout(self.dropout_rate)

    def forward(self, x):
        # Standard order: Linear → BatchNorm → ReLU → Dropout
        # BatchNorm before activation is most common pattern

        x = self.input(x)
        x = self.bn_input(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.h1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.h2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.h3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.h4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.h5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.h6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.h7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.h8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.h9(x)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layer - no batchnorm, no activation
        return self.h10(x)
