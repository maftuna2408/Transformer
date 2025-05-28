from torch import nn

class TokenEmbedding(nn.Embedding):
    """
    TokenEmbedding uses nn.Embedding to convert token indices to dense vectors.

    This layer maps each token in the vocabulary to a learnable d_model-dimensional vector.
    It serves as the initial representation of input/output tokens in the Transformer.

    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens).
        d_model (int): Dimensionality of each token embedding.
    """

    def __init__(self, vocab_size, d_model):
        # Initialize the embedding layer with padding index set to 1
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
