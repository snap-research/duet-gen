# Partially from https://github.com/Mael-zys/T2M-GPT

import torch
import torch.nn as nn
from torch import Tensor, nn
from dataset_preparation.data_utils import *
from models.vqvae.encdec import Encoder, Decoder
from models.vqvae.quantize_cnn import QuantizeEMAReset
from collections import OrderedDict


class Hierarchical_VQVAE_TwoPerson(nn.Module):

    def __init__(self,
                 input_feats: int,
                 output_feats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=1024,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 **kwargs) -> None:

        super().__init__()
        self.single_person_feats = input_feats // 2
        self.code_dim_t = code_dim * 2
        self.code_dim_b = code_dim
        self.code_num = code_num
        self.output_emb_width = output_emb_width
        self.latent_dim = (output_emb_width//2) + 200
        self.down_t = down_t
        self.stride_t = stride_t
        self.width = width
        self.depth = depth
        self.cond_drop_prob = 0.1
        
        num_genre = 10
        mfcc_dim = 40
        chroma_dim = 12
        self.twoperson_encoder = nn.Conv1d(2, 1, 3, 1, 1)
        self.encoder_b = Encoder(self.single_person_feats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)
        self.encoder_t = Encoder(output_emb_width,
                               self.code_dim_t,
                               down_t-1,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.quantize_conv_b = nn.Conv1d(2 * self.output_emb_width, self.code_dim_b, 3, 1, 1)
        self.upsample_t = nn.ConvTranspose1d(self.code_dim_t, self.code_dim_b, kernel_size=2, stride=2)
        self.mfcc_encoder_b =  Encoder(mfcc_dim,
                               self.output_emb_width//8,
                               down_t + 1,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)
        self.chroma_encoder_b = Encoder(chroma_dim,
                               self.output_emb_width//8,
                               down_t + 1,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.genre_emb_dim = kwargs['genre_emb_dim']
        self.genre_emb = nn.Linear(num_genre, self.genre_emb_dim)
        self.music_feats_enc_dim_b = (output_emb_width//8) + (output_emb_width//8)

       
        self.quantizer_t = QuantizeEMAReset(code_num, self.code_dim_t, mu=0.99, gpu_id=kwargs['gpu_id'])
        self.quantizer_b = QuantizeEMAReset(code_num, self.code_dim_b, mu=0.99, gpu_id=kwargs['gpu_id'])
        
        self.decoder_t = Decoder(self.output_emb_width,
                               self.code_dim_t,
                               down_t - 1,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)
        self.decoder_b = Decoder(output_feats,
                               self.code_dim_b + self.code_dim_b + self.music_feats_enc_dim_b + self.genre_emb_dim,
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


    def forward(self, two_person_features: Tensor, genre:Tensor, mfcc_feats: Tensor, chroma_feats:Tensor):
        B, seq_len, two_person_dim = two_person_features.shape        
        twoperson_feats_reshape = torch.cat((two_person_features[..., :self.single_person_feats].unsqueeze(-1),
                                    two_person_features[..., self.single_person_feats:].unsqueeze(-1)), -1)
        twoperson_feats_reshape = twoperson_feats_reshape.view(B*seq_len , 2, -1)
        features = self.twoperson_encoder(twoperson_feats_reshape).view(B, seq_len, -1)
        features = self.preprocess(features)
        # Top level encoding and quantization
        enc_b = self.encoder_b(features)
        encoder_t = self.encoder_t(enc_b)
        
        top_token_len = encoder_t.shape[-1]
        x_quantized_t, loss_t, perplexity_t, code_idx_t = self.quantizer_t(encoder_t)
        decoder_t  = self.decoder_t(x_quantized_t)
        upsample_t = self.upsample_t(x_quantized_t)

        # Bottom level encoding and quantization
        encoder_b = torch.cat([decoder_t, enc_b], dim=1)
        quant_b = self.quantize_conv_b(encoder_b)
        bottom_token_len = quant_b.shape[-1]
        x_quantized_b, loss_b, perplexity_b, code_idx_b = self.quantizer_b(quant_b)

        # Encoding the conditions for bottom level decoding
        randomizer = torch.randint(0, 2, (1, 1))
        if self.training and randomizer == 1:
            genre = torch.zeros_like(genre).to(features.device)
        genre_emb = self.genre_emb(genre.float().to(features.device)).reshape(len(genre), self.genre_emb_dim, 1)
        genre_emb_b = genre_emb.repeat(1, 1, bottom_token_len)
        mfcc_in_b = self.mfcc_encoder_b(self.preprocess(mfcc_feats))
        chroma_in_b = self.chroma_encoder_b(self.preprocess(chroma_feats))

        x_cond_b = torch.cat([upsample_t, x_quantized_b,  mfcc_in_b, chroma_in_b, genre_emb_b], dim=1)
        x_out = self.preprocess(self.decoder_b(x_cond_b))
        perplexity = torch.cat((perplexity_b.unsqueeze(0), perplexity_t.unsqueeze(0)))
        return x_out, (loss_b + loss_t), perplexity, [code_idx_b, code_idx_t]

    def encode(self, two_person_features: Tensor):
        B, seq_len, two_person_dim = two_person_features.shape        
        twoperson_feats_reshape = torch.cat((two_person_features[..., :self.single_person_feats].unsqueeze(-1),
                                    two_person_features[..., self.single_person_feats:].unsqueeze(-1)), -1)
        twoperson_feats_reshape = twoperson_feats_reshape.view(B*seq_len , 2, -1)
        features = self.twoperson_encoder(twoperson_feats_reshape).view(B, seq_len, -1)
        features = self.preprocess(features) 
        # Top level encoding and quantization
        enc_b = self.encoder_b(features)
        encoder_t = self.encoder_t(enc_b)
        
        x_quantized_t, loss_t, perplexity_t, code_idx_t = self.quantizer_t(encoder_t)
        decoder_t  = self.decoder_t(x_quantized_t)

        # Bottom level encoding and quantization
        encoder_b = torch.cat([decoder_t, enc_b], dim=1)
        quant_b = self.quantize_conv_b(encoder_b)
        bottom_token_len = quant_b.shape[-1]
        x_quantized_b, loss_b, perplexity_b, code_idx_b = self.quantizer_b(quant_b)
        code_idx = [code_idx_b, code_idx_t]   
        return code_idx
        
    def decode(self, code_idx: Tensor, genre:Tensor, mfcc_feats: Tensor, chroma_feats:Tensor):
        code_idx_b = code_idx[0]
        code_idx_t = code_idx[1]
        B, bottom_token_len = code_idx_b.shape
        x_quantized_t = self.quantizer_t.dequantize(code_idx_t)
        x_quantized_t = x_quantized_t.view(B, -1, self.code_dim_t).permute(0, 2, 1).contiguous()
        x_quantized_b = self.quantizer_b.dequantize(code_idx_b)
        x_quantized_b = x_quantized_b.view(B, -1, self.code_dim_b).permute(0, 2, 1).contiguous()
        upsample_t = self.upsample_t(x_quantized_t)
    
        genre = torch.zeros_like(genre).to(code_idx_b.device)
        genre_emb = self.genre_emb(genre.float().to(code_idx_b.device)).reshape(len(genre), self.genre_emb_dim, 1)
        genre_emb_b = genre_emb.repeat(1, 1, bottom_token_len)
        mfcc_in_b = self.mfcc_encoder_b(self.preprocess(mfcc_feats))
        chroma_in_b = self.chroma_encoder_b(self.preprocess(chroma_feats))
        x_cond_b = torch.cat([upsample_t, x_quantized_b,  mfcc_in_b, chroma_in_b, genre_emb_b], dim=1)
        x_out = self.preprocess(self.decoder_b(x_cond_b))
        return x_out
