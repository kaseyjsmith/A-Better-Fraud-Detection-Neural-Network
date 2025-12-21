from .base import BaseFraudNN
from torch.nn import Dropout, Linear
import torch.nn.functional as F


class DeepFraudNN(BaseFraudNN):
    """
    DeepFraudNN keeps a fairly narrow layer strucutre, but is deeper than BaseFraudNN

    The issue with deep neural networks is that they are prone to vanishing gradients.
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

        self.dropout_rate = dropout_rate
        self.dropout = Dropout(self.dropout_rate)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.dropout(x)

        x = F.relu(self.h1(x))
        x = self.dropout(x)

        x = F.relu(self.h2(x))
        x = self.dropout(x)

        x = F.relu(self.h3(x))
        x = self.dropout(x)

        x = F.relu(self.h4(x))
        x = self.dropout(x)

        x = F.relu(self.h5(x))
        x = self.dropout(x)

        x = F.relu(self.h6(x))
        x = self.dropout(x)

        x = F.relu(self.h7(x))
        x = self.dropout(x)

        x = F.relu(self.h8(x))
        x = self.dropout(x)

        x = F.relu(self.h9(x))
        x = self.dropout(x)

        return self.h10(x)
