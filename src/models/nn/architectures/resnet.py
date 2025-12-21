from .base import BaseFraudNN
from torch.nn import Dropout, Linear
import torch.nn.functional as F
import torch


class ResNetFraudNN(BaseFraudNN):
    """
    ResNet-style architecture with skip connections to solve vanishing gradients.

    Skip connections allow gradients to flow directly through the network,
    preventing the exponential decay that occurs in deep networks.

    Architecture: Similar depth to DeepFraudNN (11 layers) but with residual blocks

    NOTE: Because of the preservation of gradients early in the training cycle, ResNet
    models should _generally_ be trained through less epochs. With too many epochs
    (e.g. 50), gradients grow, weights get pushed to extreme values, loss quickly gets
    out of control, and the model collapses.
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

        # Input layer
        self.input = Linear(input, h1)

        # First residual block (64 -> 64 -> 64)
        self.h1 = Linear(h1, h2)
        self.h2 = Linear(h2, h3)

        # Second residual block (64 -> 64 -> 64)
        self.h3 = Linear(h3, h4)
        self.h4 = Linear(h4, h5)

        # Transition layer (64 -> 32) - dimension reduction
        self.h5 = Linear(h5, h6)

        # Third residual block (32 -> 32 -> 32)
        self.h6 = Linear(h6, h7)
        self.h7 = Linear(h7, h8)

        # Transition layer (32 -> 16) - dimension reduction
        self.h8 = Linear(h8, h9)

        # Final residual block (16 -> 16)
        self.h9 = Linear(h9, h10)

        # Output layer
        self.h10 = Linear(h10, out)

        self.dropout_rate = dropout_rate
        self.dropout = Dropout(self.dropout_rate)

    def forward(self, x):
        # Input layer
        x = F.relu(self.input(x))
        x = self.dropout(x)

        # First residual block: 64 -> 64 -> 64
        identity = x
        x = F.relu(self.h1(x))
        x = self.dropout(x)
        x = F.relu(self.h2(x))
        x = self.dropout(x)
        x = x + identity  # Add skipp connection

        # Second residual block: 64 -> 64 -> 64
        identity = x  # Save input for skip connection
        x = F.relu(self.h3(x))
        x = self.dropout(x)
        x = F.relu(self.h4(x))
        x = self.dropout(x)
        x = x + identity  # Add skip connection

        # Transition: 64 -> 32 (no skip connection - dimension changes)
        x = F.relu(self.h5(x))
        x = self.dropout(x)

        # Third residual block: 32 -> 32 -> 32
        identity = x  # Save input for skip connection
        x = F.relu(self.h6(x))
        x = self.dropout(x)
        x = F.relu(self.h7(x))
        x = self.dropout(x)
        x = x + identity  # Add skip connection

        # Transition: 32 -> 16 (no skip connection - dimension changes)
        x = F.relu(self.h8(x))
        x = self.dropout(x)

        # Final residual block: 16 -> 16
        identity = x  # Save input for skip connection
        x = F.relu(self.h9(x))
        x = self.dropout(x)
        x = x + identity  # Add skip connection

        # Output layer (no activation - BCEWithLogitsLoss expects logits)
        return self.h10(x)
