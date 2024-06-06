from torch import nn
import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from model.positional_encoding import FixedPositionalEncoding

class univariate_TSTransformer(nn.Module):

    def __init__(self, d_model = 512, input_window = 60, output_window = 6, nhead = 8, num_layers = 6, dropout = 0.1, max_len=1024):
        super().__init__()

        self.input_window = input_window
        self.output_window = output_window
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout

        self.temp = nn.Linear(1, d_model)

        self.encoder = nn.Sequential(
            nn.Linear(1, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(self.input_window, (self.input_window + self.output_window)//2),
            nn.ReLU(),
            nn.Linear((self.input_window+self.output_window)//2, self.output_window)
        ) 

        self.positional_encoding = FixedPositionalEncoding(d_model = self.d_model, dropout = self.dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers) 
        self.pos_encoder = FixedPositionalEncoding(d_model, dropout)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.linear(output)[:,:,0]
        output = self.linear2(output)
        return output
    
    def predict(self, src):
        src = self.encoder(src)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src.transpose(0,1)).transpose(0,1)
        output = self.linear(output)[:,:,0]
        output = self.linear2(output)
        return output


         