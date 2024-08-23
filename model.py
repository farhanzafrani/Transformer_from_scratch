import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: bool):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(seq_len, d_model).float()
        position = torch.arange(0, seq_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10e-9):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, 1))  # multiplied
        self.beta = nn.Parameter(torch.zeros(1, 1, 1))  # added

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: bool):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(
                attention_scores
            )  # (batch_size, h, seq_len, seq_len)
        context = torch.matmul(attention_scores, value)
        return context, attention_scores

    def forward(self, q, k, v, mask):
        query = self.Wq(q)  # shape (batch, seq_len, d_model)
        key = self.Wk(k)
        value = self.Wv(v)

        # Split query, key, value into heads
        # shape (batch, seq_len, d_model) --> (batch, seq_len, n_heads, d_k) --> (batch, n_heads, seq_len, d_k)
        query = query.view(
            query.size(0), query.size(1), self.n_heads, self.d_k
        ).permute(0, 2, 1, 3)
        key = key.view(key.size(0), key.size(1), self.n_heads, self.d_k).permute(
            0, 2, 1, 3
        )
        value = value.view(
            value.size(0), value.size(1), self.n_heads, self.d_k
        ).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        x, self.attention_score = MultiHeadAttentionBlock.attention(
            query, key, value, self.dropout
        )
        # (batch, n_heads, seq_len, d_k) --> (batch, seq_len, n_heads, d_k)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_k * self.n_heads)
        return self.Wo(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: bool):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.norm(x)))  # !WARNING: Will look into it.


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout=dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):

        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, lambda x: self.feed_forward_block(x))
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout=dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, lambda x: self.feed_forward_block(x))
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, tgt_mask, src_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x, dim=-1))


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        tgt_embedding: InputEmbedding,
        src_embedding: InputEmbedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embedding = tgt_embedding
        self.src_embedding = src_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encoder(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decoder(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def projection(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_vocab: int,
    tgt_vocab: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
):
    # create embedding layer
    src_emb = InputEmbedding(d_model, src_vocab)
    tgt_emb = InputEmbedding(d_model, tgt_vocab)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    # create encoder block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_blcok = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention_block, feed_forward_blcok, dropout
        )
        encoder_blocks.append(encoder_block)
    # create decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab)

    # create transformer
    transformer = Transformer(
        encoder, decoder, tgt_emb, src_emb, src_pos, tgt_pos, projection_layer
    )

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
