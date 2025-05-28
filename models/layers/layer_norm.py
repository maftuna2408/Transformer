import torch
from torch import nn

class LayerNorm(nn.Module):
    """
    Applies layer normalization over the last dimension of the input.

    Used in the Transformer architecture to stabilize and accelerate training.
    This layer normalizes input values across the embedding dimension.

    Args:
        d_model (int): Size of the last dimension (i.e., embedding size).
        eps (float): Small constant to prevent division by zero.
    """
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.gamma * normalized + self.beta

    