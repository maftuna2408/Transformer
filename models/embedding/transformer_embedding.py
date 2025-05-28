from torch import nn
from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding

class TransformerEmbedding(nn.Module):
    """
   TransformerEmbedding(x) = Dropout(TokenEmbedding(x) + PositionalEncoding(x))

    This module adds positional information to token representations
    and applies dropout to improve generalization.

    Args:
        vocab_size (int): Vocabulary size.
        d_model (int): Embedding and model dimension.
        max_len (int): Maximum sequence length.
        drop_prob (float): Dropout probability.
        device (torch.device): Device for tensor initialization.
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        Forward pass of the embedding layer.

        Args:
            x (Tensor): Input token indices of shape [batch_size, seq_len]
        Returns:
            Tensor: Embedded tokens with positional information [batch_size, seq_len, d_model]
        """
        tok_emb = self.tok_emb(x)       # [batch_size, seq_len, d_model]
        pos_emb = self.pos_emb(x)       # [batch_size, seq_len, d_model]
        return self.drop_out(tok_emb + pos_emb)
