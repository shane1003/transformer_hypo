import torch.nn as nn
import torch
import numpy as np

class FixedPositionalEncoding(nn.Module):
    """
    Positional encoding of input
    Args:
        d_model: the embed dimension
        dropout: the dropout value
        max_len: the max length of the incoming sequence
    """

    def __init__(self, d_model, max_len=5000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x