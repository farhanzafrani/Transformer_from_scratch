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
