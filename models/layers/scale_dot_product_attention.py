import torch
from torch import nn
import math

class ScaleDotProductAttention(nn.Module):
    """
    Computes scaled dot-product attention.

    Query, Key, and Value are 3 matrices used to calculate attention weights.
    This is the core operation in self-attention and cross-attention.

    Formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        d_tensor = k.size(-1)
        k_t = k.transpose(-2, -1)
        score = torch.matmul(q, k_t) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        attention = self.softmax(score)
        out = torch.matmul(attention, v)

        return out, attention
