from torch import nn
from models.model.encoder import Encoder
from models.model.decoder import Decoder

class Transformer(nn.Module):
    """
    Full Transformer Model: Encoder + Decoder + Masking
    """
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx,
                 enc_vocab_size, dec_vocab_size,
                 d_model, n_head, max_len, ffn_hidden,
                 n_layers, drop_prob, device):
        super().__init__()

        # Special token indices
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx

        self.device = device

        # Encoder module
        self.encoder = Encoder(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            enc_vocab_size=enc_vocab_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            device=device
        )

        # Decoder module
        self.decoder = Decoder(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            dec_vocab_size=dec_vocab_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            device=device
        )

    def forward(self, src, trg):
        """
        src: [batch_size, src_len]
        trg: [batch_size, trg_len]
        """
        # Generate source and target masks
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # Encode source sentence
        enc_src = self.encoder(src, src_mask)

        # Decode using encoder output and target input
        output = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output

    def make_src_mask(self, src):
        # Mask pad tokens in source sentence
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask  # [batch_size, 1, 1, src_len]

    def make_trg_mask(self, trg):
        # Padding mask
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # Causal (look-ahead) mask
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # Combine masks
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask  # [batch_size, 1, trg_len, trg_len]
