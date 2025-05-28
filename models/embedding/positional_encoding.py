import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements fixed sinusoidal positional encoding as proposed by Vaswani et al. (2017).

    This layer injects information about the position of each token in the sequence,
    since Transformers have no recurrence or convolution.

    Args:
        d_model (int): Dimensionality of embeddings.
        max_len (int): Maximum length of the input sequences.
        device (torch.device): Device to initialize positional encodings on.
    """

    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        # Precompute the positional encodings matrix [max_len, d_model]
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # static encoding, not learnable

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor: Positional encoded tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        return self.encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
