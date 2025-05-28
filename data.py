from util.data_loader import make_loader
from conf import batch_size, max_len, device

# === Dataset paths ===
train_loader, valid_iter, test_iter, SRC_VOCAB, TRG_VOCAB = make_loader(
    src_path='data/English.txt',
    trg_path='data/Turkish.txt',
    batch_size=batch_size,
    max_len=max_len,
    device=device
)


# === Indexes ===
src_pad_idx = SRC_VOCAB['<pad>']
trg_pad_idx = TRG_VOCAB['<pad>']
trg_sos_idx = TRG_VOCAB['<sos>']

# === Vocabulary sizes ===
enc_vocab_size = len(SRC_VOCAB)
dec_vocab_size = len(TRG_VOCAB)

# === Export all to train.py ===
# data.py (at bottom)
__all__ = [
    "train_loader", "valid_iter", "test_iter",
    "src_pad_idx", "trg_pad_idx", "trg_sos_idx",
    "enc_vocab_size", "dec_vocab_size",
    "SRC_VOCAB", "TRG_VOCAB"  
]