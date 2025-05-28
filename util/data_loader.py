from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch

class TranslationDataset(Dataset):
    def __init__(self, src_lines, trg_lines, src_vocab, trg_vocab, max_len=100):
        self.src_lines = src_lines
        self.trg_lines = trg_lines
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src = self.encode(self.src_lines[idx], self.src_vocab)
        trg = self.encode(self.trg_lines[idx], self.trg_vocab)
        return torch.tensor(src), torch.tensor(trg)

    def encode(self, sentence, vocab):
        tokens = sentence.strip().lower().split()
        ids = [vocab['<sos>']] + [vocab.get(tok, vocab['<unk>']) for tok in tokens[:self.max_len-2]] + [vocab['<eos>']]
        return ids + [vocab['<pad>']] * (self.max_len - len(ids))

def load_lines(path):
    with open(path, encoding='utf-8') as f:
        return f.readlines()

def build_vocab(lines, min_freq=2):
    counter = Counter()
    for line in lines:
        counter.update(line.strip().lower().split())
    
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def make_loader(src_path, trg_path, batch_size, max_len, device):
    src_lines = load_lines(src_path)
    trg_lines = load_lines(trg_path)

    assert len(src_lines) == len(trg_lines), "Line count mismatch between English and Turkish!"

    src_vocab = build_vocab(src_lines)
    trg_vocab = build_vocab(trg_lines)

    dataset = TranslationDataset(src_lines, trg_lines, src_vocab, trg_vocab, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return loader, src_vocab, trg_vocab
