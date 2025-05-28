import spacy

class Tokenizer:
    def __init__(self):
        self.spacy_en = spacy.load("en_core_web_sm")
        self.spacy_tr = spacy.blank("tr")  # Blank Turkish tokenizer

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def tokenize_tr(self, text):
        return [tok.text.lower() for tok in self.spacy_tr.tokenizer(text)]
