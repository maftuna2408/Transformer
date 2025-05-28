from torch import nn
from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):
    """
    The Transformer encoder module.
    Applies embedding + N layers of EncoderLayer
    """
    def __init__(self, enc_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        # Embedding layer with positional encoding
        self.emb = TransformerEmbedding(
            d_model=d_model,
            max_len=max_len,
            vocab_size=enc_vocab_size,
            drop_prob=drop_prob,
            device=device
        )

        # Stack of EncoderLayers
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                drop_prob=drop_prob
            ) for _ in range(n_layers)
        ])

    def forward(self, x, src_mask):
        """
        x: [batch_size, src_len]
        src_mask: [batch_size, 1, 1, src_len]
        """
        # Convert tokens into embeddings with position info
        x = self.emb(x)

        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, src_mask)

        return x  # Output: [batch_size, src_len, d_model]

