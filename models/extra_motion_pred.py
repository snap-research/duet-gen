# Partially from https://github.com/Mael-zys/T2M-GPT

from typing import List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from dataset_preparation.data_utils import *
from models.vqvae.encdec import Encoder, Decoder
from collections import OrderedDict

class Global_Trajectory_Pred(nn.Module):

    def __init__(self,
                 input_feats = 265,
                 output_feats = 6,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 **kwargs) -> None:

        super().__init__()
        self.twoperson_encoder = nn.Conv1d(2, 1, 3, 1, 1)
        self.encoder = Encoder(input_feats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        
        self.decoder = Decoder(output_feats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, two_person_features: Tensor):
        # Preprocess
        B, seq_len, dim, _ = two_person_features.shape    
        twoperson_feats_reshape = two_person_features.view(B*seq_len , 2, -1)
        features = self.twoperson_encoder(twoperson_feats_reshape).view(B, seq_len, -1)

        x_in = self.preprocess(features)
        
        
        # Encode
        x_encoder = self.encoder(x_in)
        # decoder
        x_decoder = self.decoder(x_encoder)
        x_out = self.postprocess(x_decoder)
        return x_out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class TransAE(nn.Module): 
    def __init__(self, nfeats=3, latent_dim=150,
                 hidden_dim=128, ff_size=512, num_layers=2, num_heads=2, dropout=0.05,
                 ablation=None, activation="gelu"):
        super(TransAE, self).__init__()
        self.nfeats = nfeats
        self.hidden_dim = hidden_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.input_feats = self.nfeats
        self.skelEmbedding = nn.Linear(self.input_feats, self.hidden_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.hidden_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)
        
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.hidden_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)      
        self.finallayer = nn.Linear(self.hidden_dim, self.input_feats)



    def forward(self, input_pose):
        bs = input_pose.shape[0]
        T = input_pose.shape[1]
        # lengths = torch.ones((bs, 1)) * T
        lengths = [T] * bs
        mask = torch.ones(bs, T).type(torch.bool).to(input_pose.device)
        x = input_pose.reshape(bs, T, -1).permute((1, 0, 2))
        x_ = self.skelEmbedding(x)
       
        x1 = self.sequence_pos_encoder(x_)
        z = self.seqTransEncoder(x1, src_key_padding_mask=~mask)
        timequeries_ = torch.zeros(T, bs, self.hidden_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries_)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        output = self.finallayer(output).reshape(T, bs, -1)
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 0, 2)
        
        return output

    def encode(self, pose):
        bs = pose.shape[0]
        T = pose.shape[1]
        lengths = [T] * bs
        mask = torch.ones(bs, T).type(torch.bool).to(pose.device)
        x = pose.reshape(bs, T, -1).permute((1, 0, 2))
        x_ = self.skelEmbedding(x)
        x1 = self.sequence_pos_encoder(x_)
        z = self.seqTransEncoder(x1, src_key_padding_mask=~mask)
        return z.permute(1, 0, 2)