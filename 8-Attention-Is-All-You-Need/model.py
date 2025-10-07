import math
import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    pass


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:  # (d_model, vocab_size)
        super().__init__()  # Initialize the parent class
        self.d_model = d_model  # Dimension of the model
        self.vocab_size = vocab_size  # Size of the vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model)  # Embedding layer

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()  # Initialize the parent class
        self.d_model = d_model  # Dimension of the model
        self.seq_len = seq_len  # Maximum sequence length
        self.dropout = nn.Dropout(dropout)  # Dropout layer

        """Create a positional encoding matrix of shape (seq_len, d_model)
        We are using positional encoding to give the model some information about the relative position of the words in the sentence.
        The positional encoding is taken from the paper "Attention is All You Need" where we use sine and cosine functions of different frequencies to encode the positions. The sine function is used for even indices and the cosine function is used for odd indices so that each dimension of the positional encoding corresponds to a sinusoid of different wavelength.
        """
        pe = torch.zeros(
            seq_len, d_model)  # Initialize the positional encoding matrix
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1)  # Create a column vector of positions (seq_len, 1)
        # Create the div_term for the sine and cosine functions
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add a batch dimension (1, seq_len, d_model)
        # Register pe as a buffer so it is not considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input embeddings
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        # Apply dropout to the sum of embeddings and positional encodings
        return self.dropout(x)


class LayerNormalisation(nn.Module):

    # where eps is epsilon a small value to avoid division by zero
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()  # Initialize the parent class
        self.eps = eps  # Epsilon value to avoid division by zero
        self.alpha = nn.Parameter(torch.ones(1))  # Scale parameter
        self.bias = nn.Parameter(torch.zeros(1))  # Shift parameter

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # Mean of the last dimension
        # Standard deviation of the last dimension
        std = x.std(dim=-1, keepdim=True)
        # Normalize and scaling
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()  # Initialize the parent class
        self.linear1 = nn.Linear(d_model, d_ff)  # First linear layer
        self.linear2 = nn.Linear(d_ff, d_model)  # Second linear layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    """
    d_model: Dimension of the model
    h: Number of heads
    dropout: Dropout rate
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()  # Initialize the parent class
        self.d_model = d_model  # Dimension of the model
        self.h = h  # Number of heads
        # d_model must be divisible by h because we will split the d_model into h heads
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h  # Dimension of each head
        # Linear layer for query
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        # Linear layer for key
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        # Linear layer for value
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        # Linear layer for output
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)  # Dropout layer

    @staticmethod  # Compute scaled dot-product attention
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)  # last value of query key and value is d_k
        # (batch, h, seq_len, d_k) x (batch, h, d_k, seq_len) --> (batch, h, seq_len, seq_len)

        # applying first par of the attention formula
        attention_scores = torch.matmul(
            query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                mask == 0, -1e9)  # Mask out the padding tokens
        # Apply softmax to get attention weights
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)  # Apply dropout

        # (batch, h, seq_len, d_k)
        return torch.matmul(attention_scores, value), attention_scores

    def forward(self, q, k, v, mask):
        # (batch, seq_len, d_model)-> (batch, seq_len, d_model)
        query = self.w_q(q)
        # (batch, seq_len, d_model)-> (batch, seq_len, d_model)
        key = self.w_k(k)
        # (batch, seq_len, d_model)-> (batch, seq_len, d_model)
        value = self.w_v(v)

        # splitting the d_model into h heads
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1],
                       self.h, self.d_k).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Apply attention on all the projected vectors in batch
        # (batch, h, seq_len, d_k), (batch, h, seq_len, seq_len), (batch, h, seq_len, d_k)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(
            x.shape[0], -1, self.h * self.d_k)
        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # Multi-head attention block
        self.feed_forward_block = feed_forward_block  # Feed-forward block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)])  # Two residual connections for the two sub-layers in the encoder block

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask))  # Self-attention sub-layer passing the same x as q, k, v
        x = self.residual_connections[1](
            x, self.feed_forward_block)  # Feed-forward sub-layer
        return x


class Encoder(nn.Module):

    # nn.ModuleList[EncoderBlock]
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()  # Initialize the parent class
        self.layers = layers  # List of encoder blocks
        self.norm = LayerNormalization(features)  # Final layer normalization

    def forward(self, x, mask):
        for layer in self.layers:  # Iterate through each encoder block
            # Pass the output of the previous block as input to the next block
            x = layer(x, mask)
        return self.norm(x)  # Apply final layer normalization


class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:

        super().__init__()  # Initialize the parent class
        # Masked multi-head attention block
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block  # Multi-head attention block
        self.feed_forward_block = feed_forward_block  # Feed-forward block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)])  # Three residual connections for the three sub-layers in the decoder block

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask))  # Masked self-attention sub-layer
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(
            x, encoder_output, encoder_output, src_mask))  # Cross-attention sub-layer
        x = self.residual_connections[2](
            x, self.feed_forward_block)  # Feed-forward sub-layer
        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # List of decoder blocks
        self.norm = LayerNormalization(features)  # Final layer normalization

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            # Pass the output of the previous block as input to the next block
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)  # Apply final layer normalization


# building projection layer for projecting the embedding into vocabulary
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        # Linear layer to project the d_model to vocab_size
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):

    # (encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()  # Initialize the parent class
        self.encoder = encoder  # Encoder
        self.decoder = decoder  # Decoder
        self.src_embed = src_embed  # Source embeddings
        self.tgt_embed = tgt_embed  # Target embeddings
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos  # Positional encodings
        self.projection_layer = projection_layer  # Projection layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)  # Apply source embeddings
        src = self.src_pos(src)  # Add positional encodings
        return self.encoder(src, src_mask)  # Pass through the encoder

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)  # Apply target embeddings
        tgt = self.tgt_pos(tgt)  # Add positional encodings
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # Project the decoder output to vocabulary size
        return self.projection_layer(x)


"""
Here we have a function to build the entire transformer model by creating the encoder and decoder blocks, multi-head attention blocks, feed-forward blocks, embeddings, positional encodings, and projection layer.
1. src_vocab_size: Size of the source vocabulary
2. tgt_vocab_size: Size of the target vocabulary
3. src_seq_len: Maximum sequence length for the source
4. tgt_seq_len: Maximum sequence length for the target
5. d_model: Dimension of the model (default: 512)
6. N: Number of layers in the encoder and decoder (default: 6)
7. h: Number of heads in the multi-head attention (default: 8)
8. dropout: Dropout rate (default: 0.1)
9. d_ff: Dimension of the feed-forward network (default: 2048)
"""


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # creating the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # creating the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # creating the encoder blocks
    encoder_blocks = []

    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout)  # creating the feed forward block
        feed_forward_block = FeedForwardBlock(
            d_model, d_ff, dropout)  # creating the encoder block
        encoder_block = EncoderBlock(
            d_model, encoder_self_attention_block, feed_forward_block, dropout)  # adding the encoder block to the list
        encoder_blocks.append(encoder_block)  # creating the encoder

    # creating the decoder blocks
    decocer_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout)  # creating the feed forward block
        decoder_cross_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout)  # creating the feed forward block
        feed_forward_block = FeedForwardBlock(
            d_model, d_ff, dropout)  # creating the decoder block
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)  # adding the decoder block to the list
        decocer_blocks.append(decoder_block)  # creating the decoder

    # creating the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decocer_blocks))

    # creating the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # creating the transformer model
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialising the parameters of the model
    for p in transformer.parameters():
        if p.dim() > 1:
            # Xavier uniform initialization for weights with more than 1 dimension
            nn.init.xavier_uniform_(p)

    return transformer
