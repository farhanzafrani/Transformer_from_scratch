import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: bool):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.zeros(seq_len, d_model).float()
        position = torch.arange(0, seq_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # shape (1, seq_len, d_model)
        self.register_buffer("pe", self.pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.shape(1), :]).requires_grad_(False)


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


class MultiHeadAttention(nn.Module):
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
        x, self.attention_score = MultiHeadAttention.attention(
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
        self_attention_block: MultiHeadAttention,
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
