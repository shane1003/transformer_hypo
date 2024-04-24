from torch import nn, Tensor
import torch
import torch.optim as optim
from torch.nn import Transformer
import numpy as np
import random
import math

class Transformer(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()

        self.transformer = Transformer(ntoken=ntoken, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers, dropout=dropout)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.pos_encoder_d = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.encoder = nn.Linear(ntoken, d_model)
        self.encoder_d = nn.Linear(ntoken, d_model)
        self.linear = nn.Linear(d_model, 1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.bias.data.zero_()
        self.encoder_d.bias.data.zero_()
        self.encoder_d.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ''[seq_len, batch_size]''
            src_mask: Tensor, shape ''[seq_len, seq_len]''

        Returns:
            output Tensor of shape ''[seq_len, batch_size, ntoken]''
        """

        src = self.encoder(src)
        src = self.pos_encoder(src)

        tgt = self.encoder_d(tgt)
        tgt = self.pos_encoder_d(tgt)

        output = self.transformer(src.transpose(0,1), tgt.transpose(0,1), src_mask, tgt_mask)
        output = self.linear(output)

        return output

    def generate_square_subusequent_mask(sz: int) -> Tensor:
        """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model ))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ''[seq_len, batch_size, embedding_dim]''
        """

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)