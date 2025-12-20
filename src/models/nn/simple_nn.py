"""
First pass neural network building.
"""

from torch.nn import Module, BCELoss, Linear, Dropout
import torch.nn.functional as F
from torch.optim import Adam

from uuid import uuid4


class FraudDetectionNN(Module):
    def __init__(self, input=30, h1=64, h2=32, h3=16, h4=8, out=1, dropout_rate=0.2):
        super().__init__()
        self.input = Linear(input, h1)
        self.h1 = Linear(h1, h2)
        self.h2 = Linear(h2, h3)
        self.h3 = Linear(h3, h4)
        self.h4 = Linear(h4, out)

        # Implement dropout to not have the model not rely on a frequent path
        self.dropout_rate = dropout_rate
        self.dropout = Dropout(self.dropout_rate)

        self.run_id = uuid4()

    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.dropout(x)

        x = F.relu(self.h1(x))
        x = self.dropout(x)

        x = F.relu(self.h2(x))
        x = self.dropout(x)

        x = F.relu(self.h3(x))
        x = self.dropout(x)

        return self.h4(x)
