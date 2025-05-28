from torch import nn
from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
    """
    Transformer decoder module.
    Embedding + N DecoderLayers + output projection
    """
    def __init__(self, dec_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        # Embedding for target language (with position info)
        self.emb = TransformerEmbedding(
            d_model=d_model,
            drop_prob=drop_prob,
            max_len=max_len,
            vocab_size=dec_vocab_size,
            device=device
        )

        # Stack of DecoderLayers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                drop_prob=drop_prob
            ) for _ in range(n_layers)
        ])

        # Linear layer to map model output to vocab size
        self.linear = nn.Linear(d_model, dec_vocab_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        trg: [batch_size, trg_len]
        enc_src: [batch_size, src_len, d_model]
        trg_mask: for causal masking in decoder
        src_mask: for padding in encoder output
        """
        # Step 1: Embed target input
        trg = self.emb(trg)

        # Step 2: Pass through decoder layers
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # Step 3: Predict next word probabilities
        output = self.linear(trg)

        return output  # [batch_size, trg_len, dec_vocab_size]
