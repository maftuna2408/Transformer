from collections import Counter
from torchtext.vocab import Vocab

def build_vocab(sentences, tokenizer, min_freq=2):
    counter = Counter()
    for sentence in sentences:
        counter.update(tokenizer(sentence))
    return Vocab(counter, specials=['<pad>', '<sos>', '<eos>', '<unk>'], min_freq=min_freq)
