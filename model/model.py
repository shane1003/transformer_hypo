from torch import nn, Tensor
import torch
import torch.optim as optim
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import functional as F
import numpy as np
import random
import math

class Transformer(nn.Module):

    def __init__(self, d_model: int, nhead: int, nlayers: int, dropout: float = 0.5):
        super().__init__()

        #INFO
        self.model_type = "Transformer"
        self.dim_model = d_model
        self.src_mask = None

        #LAYERS
        self.positional_encoder = PositionalEncoding(dim_model=d_model, dropout_p=dropout, max_len=5000)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.02
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src: Tensor) -> Tensor:
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.positional_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def predict(self, input, target_len):
        outputs = torch.zeros(1, target_len, 1)

        if input.shape[-1] == 1:
            input = input.unsqueeze(0)
            input = input.unsqueeze(2)
        else:
            input = input.squeeze().unsqueeze(0)
        self.eval()
        batch_size = input.shape[0]

        for t in range(target_len):
            if len(input.shape) == 2:
                input = input.unsqueeze(2)
            
            out = self.forward(input)

            if t < target_len - 1 : 
                input[:, t + 1, 0] = out[:, t , 0]

            outputs[:, t, 0] = out[:, t, 0]

        return outputs.detach().numpy().squeeze()

class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
        
        (pos, 2i) = sin(pos/10000^(2i/d_model))
        (pos, 2i+1) = cos(pos/10000^(2i/d_model))
        
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, dim_model, dropout_p, max_len=1024):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model )

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x: Tensor):
        #Residual connection + position encoding
        x = x + self.pos_encoding[:x.size(0), :]
        return self.dropout(x)
    
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        self.pos_encoding = nn.Parameter(torch.empty(max_len, 1, dim_model))
        nn.init.uniform_(self.pos_encoding, -0.02, 0.02)

    def forward(self, x: Tensor):

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class TransformerEncoder(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout_p=0.1, freeze=False):
        super(TransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.positional_encoder = LearnablePositionalEncoding(dim_model=d_model, dropout_p=dropout_p*(1.0 - freeze), max_len=max_len)
        
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=self.n_heads, dim_feedforward=dim_feedforward, dropout=dropout_p*(1.0 - freeze), activation=F.gelu)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, 1)

        self.activation = F.gelu

        self.dropout = nn.Dropout(dropout_p)

        self.feat_dim = feat_dim


    def forward(self, X, padding_masks):

        inp = X.permute(1,0,2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        inp = self.positional_encoder(inp)

        out = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
        out = self.activation(out)
        out = out.permute(1, 0, 2) # (batch size, seq_length, d_model)
        out = self.dropout(out)

        out = self.output_layer(out)

        return out