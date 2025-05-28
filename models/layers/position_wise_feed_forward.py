from torch import nn
class PositionWiseFeedForward(nn.Module):
    """
    Applies a two-layer feedforward neural network to each position independently.

    Args:
        d_model (int): Input and output dimensionality.
        hidden (int): Inner-layer hidden size.
        drop_prob (float): Dropout rate.
    """
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.linear2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
