import torch
from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism as defined in the Transformer.

    Projects Q, K, V into multiple heads, performs scaled dot-product attention,
    and then concatenates the results into a single output.

    Args:
        d_model (int): Input and output dimensionality.
        n_head (int): Number of attention heads.
    """
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.attention = ScaleDotProductAttention()

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q = self.split(self.w_q(q))
        k = self.split(self.w_k(k))
        v = self.split(self.w_v(v))

        out, attn = self.attention(q, k, v, mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, x):
        b, l, d = x.size()
        h = self.n_head
        d_h = d // h
        x = x.view(b, l, h, d_h).transpose(1, 2)  # [batch, head, seq_len, d_tensor]
        return x

    def concat(self, x):
        b, h, l, d = x.size()
        x = x.transpose(1, 2).contiguous().view(b, l, h * d)
        return x

    
