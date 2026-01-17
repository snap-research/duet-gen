# Partially from https://github.com/Mael-zys/T2M-GPT

from typing import List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from dataset_preparation.data_utils import *
from models.vqvae.encdec import Encoder, Decoder

from collections import OrderedDict
from models.transformer.transformer_tools import *
from torch.distributions.categorical import Categorical

class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim, activation):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        if activation == "relu":
            self.activation = nn.ReLU()
            
        elif activation == "silu":
            self.activation = nn.SiLU()
            
        elif activation == "gelu":
            self.activation = nn.GELU()

        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [bs, seq_len, out_feats]
        output = output.permute(0, 2, 1)  # [bs, c, seqlen]
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class Music2Motion_combined(nn.Module):
    def __init__(self,
                 input_feats: int,
                 num_tokens=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 latent_dim=512,
                 ff_size=1024, 
                 num_layers=8,
                 num_heads=8, 
                 dropout=0.1, 
                 music_feats_dim=512, 
                 cond_drop_prob=0.1,
                 **kwargs) -> None:

        super().__init__()
        num_genre = kwargs['num_genres']
        self.input_feats = input_feats 
        self.latent_dim = latent_dim
        self.code_dim = code_dim
        self.num_tokens = num_tokens
        self.dropout = dropout
        self.activation = activation
        self.norm = norm
        self.cond_drop_prob = cond_drop_prob
        self.genre_emb_dim = kwargs['genre_emb_dim']
        mfcc_dim = 40
        chroma_dim = 12
        self.mfcc_encoder = Encoder(mfcc_dim,
                               output_emb_width//4,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)
        self.chroma_encoder = Encoder(chroma_dim,
                               output_emb_width//4,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)
        self.genre_emb = nn.Linear(num_genre, self.genre_emb_dim)
        self.music_feats_enc_dim = (output_emb_width//4) + (output_emb_width//4)
        self.position_enc_ = PositionalEncoding(self.latent_dim, self.dropout, max_len=3000)
        self.mask_id = self.num_tokens
        self.token_emb = nn.Embedding(self.num_tokens+1, self.code_dim)
        self.token_lin = nn.Linear(self.code_dim, self.latent_dim)
        self.cond_lin = nn.Linear(self.music_feats_enc_dim, self.latent_dim)
        
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation=self.activation,
                                                          batch_first=True)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)
        
        self.output_process = OutputProcess(out_feats=self.num_tokens, 
                                            latent_dim=latent_dim, 
                                            activation=self.activation)
        if kwargs['noise_schedule'] == 'cosine':
            self.noise_schedule = cosine_schedule
        elif kwargs['noise_schedule'] == 'scaled_cosine':
            self.noise_schedule = scale_cosine_schedule
        elif kwargs['noise_schedule'] == 'q':
            self.noise_schedule = q_schedule
        else:   self.noise_schedule = linear_schedule
       
    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def mask_cond(self, cond, force_mask=False):   
        bs, t, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, t), device=cond.device) * self.cond_drop_prob).view(bs, t, 1)
            return cond * (1. - mask)
        else:
            return cond
    
    def internal_forward(self, x_ids: Tensor, mfcc_feats: Tensor, chroma_feats: Tensor, force_mask=False):
        B, token_len = x_ids.shape
        mfcc_in = self.mfcc_encoder(self.preprocess(mfcc_feats))
        chroma_in = self.chroma_encoder(self.preprocess(chroma_feats))
        x_cond = self.preprocess(torch.cat((mfcc_in, chroma_in), dim=1))
        x_cond = self.cond_lin(self.mask_cond(x_cond, force_mask=force_mask))

        code_idx_emb = self.token_emb(x_ids)
        token_embeddings = self.token_lin(code_idx_emb)
        seq = torch.cat((x_cond, token_embeddings), dim=1)
        trans_enc_input_seq = self.position_enc_(seq)
        trans_enc_output_seq = self.seqTransEncoder(trans_enc_input_seq)
        logits = self.output_process(trans_enc_output_seq)
        return logits[..., -token_len:]
    
    def forward(self, code_idx: Tensor, mfcc_feats: Tensor, chroma_feats: Tensor, iteration, bert_masking_scheme=True):
        B, num_tokens = code_idx.shape
        x_ids = code_idx.clone()
        # r2 = iteration/45000 if iteration/45000 < 1.0 else 1.0
        rand_time = uniform((B,), device=x_ids.device, r1=0, r2=1.0)  #random hard-code
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (num_tokens * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((B, num_tokens), device=x_ids.device).argsort(dim=-1)
        mask_token = batch_randperm < num_token_masked.unsqueeze(-1)
        masked_labels = torch.where(mask_token, code_idx, self.mask_id)

        if bert_masking_scheme and iteration > 5000:
            # Further Apply Bert Masking Scheme
            # Step 1: 10% replace with an incorrect token
            mask_rid = get_mask_subset_prob(mask_token, 0.1)
            rand_id = torch.randint_like(x_ids, high=self.num_tokens)
            x_ids = torch.where(mask_rid, rand_id, x_ids)
            # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
            mask_mid = get_mask_subset_prob(mask_token & ~mask_rid, 0.88)
            x_ids = torch.where(mask_mid, self.mask_id, x_ids)        
        else:
            x_ids = torch.where(mask_token, self.mask_id, x_ids)
            
        logits = self.internal_forward(x_ids, mfcc_feats, chroma_feats, force_mask=False)
        
        masked_ce_loss, pred_idx, acc = calc_performance(logits, masked_labels, ignore_index=self.mask_id) 

        # Trying this out because validation loss doesn't decrease
        # ce_weights = torch.ones(self.num_tokens).to(code_idx.device) * 0.5
        # mean_proximity_loss = weightedSequenceCrossEntropyLoss(logits, code_idx, ce_weights, window_size=7)
            
        return pred_idx, masked_ce_loss, acc
    
    def forward_with_cond_scale(self, x_ids: Tensor, mfcc_feats: Tensor, chroma_feats: Tensor,  cond_scale=3, force_mask=False):
        # bs = motion_ids.shape[0]
        # if cond_scale == 1:
        if force_mask:
            return self.internal_forward(x_ids, mfcc_feats, chroma_feats, force_mask=True)

        logits = self.internal_forward(x_ids, mfcc_feats, chroma_feats)
        if cond_scale == 1:
            return logits

        aux_logits = self.internal_forward(x_ids, mfcc_feats, chroma_feats, force_mask=True)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self, mfcc_feats: Tensor, chroma_feats: Tensor, token_len, timesteps=20):
        
        starting_temperature = 1
        topk_filter_thres = 0.9
        cond_scale = 3
        gsample = True
        m_lens = torch.tensor([token_len]).to(mfcc_feats.device)
        padding_mask = ~lengths_to_mask(m_lens, token_len)
        # Start from all tokens being masked
        x_ids = torch.where(padding_mask, self.mask_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.)
        
        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=mfcc_feats.device), reversed(range(timesteps))):
            # 0 < timestep < 1
            # rand_time = uniform((B,), device=x_cond.device, r1=0, r2=1.0)  # Tensor
            rand_mask_prob = self.noise_schedule(timestep)
            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = (token_len * rand_mask_prob).round().clamp(min=1)

            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            x_ids = torch.where(is_mask, self.mask_id, x_ids)

            logits = self.forward_with_cond_scale(x_ids, mfcc_feats, chroma_feats, cond_scale=cond_scale)
            logits = logits.permute(0, 2, 1)   # (b, seqlen, ntoken)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            # else:
            # temperature = starting_temperature * (steps_until_x0 / timesteps)
            # temperature = max(temperature, 1e-4)
            # print(filtered_logits.shape)
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                
                probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)

            # print(pred_ids.max(), pred_ids.min())
            # if pred_ids.
            x_ids = torch.where(is_mask, pred_ids, x_ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        x_ids = torch.where(padding_mask, -1, x_ids)
        # print("Final", ids.max(), ids.min())
        return x_ids

    @torch.no_grad()
    @eval_decorator
    def generate_edit(self, mfcc_feats: Tensor, chroma_feats: Tensor, token_len, timesteps=20):
        
        starting_temperature = 1
        topk_filter_thres = 0.9
        cond_scale = 3
        gsample = True
        m_lens = torch.tensor([token_len]).to(mfcc_feats.device)
        padding_mask = ~lengths_to_mask(m_lens, token_len)
        edit_mask = padding_mask.clone()
        edit_mask[:, 0:10] = True
        edit_mask[:, -10:] = True
        x_ids = torch.where(edit_mask, self.mask_id, self.mask_id)
        scores = torch.where(edit_mask, 1e5, 0.)
        
        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=mfcc_feats.device), reversed(range(timesteps))):
            # 0 < timestep < 1
            # rand_time = uniform((B,), device=x_cond.device, r1=0, r2=1.0)  # Tensor
            rand_mask_prob = self.noise_schedule(timestep)
            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = (token_len * rand_mask_prob).round().clamp(min=1)

            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            x_ids = torch.where(is_mask, self.mask_id, x_ids)

            logits = self.forward_with_cond_scale(x_ids, mfcc_feats, chroma_feats, cond_scale=cond_scale)
            logits = logits.permute(0, 2, 1)   # (b, seqlen, ntoken)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            # else:
            # temperature = starting_temperature * (steps_until_x0 / timesteps)
            # temperature = max(temperature, 1e-4)
            # print(filtered_logits.shape)
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                
                probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)

            # print(pred_ids.max(), pred_ids.min())
            # if pred_ids.
            x_ids = torch.where(is_mask, pred_ids, x_ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        # x_ids = torch.where(padding_mask, -1, x_ids)
        # print("Final", ids.max(), ids.min())
        return x_ids


class Music2Motion_combined_hier_bottom(nn.Module):
    def __init__(self,
                 input_feats: int,
                 num_tokens=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 latent_dim=512,
                 ff_size=1024, 
                 num_layers=8,
                 num_heads=8, 
                 dropout=0.1, 
                 music_feats_dim=512, 
                 cond_drop_prob=0.1,
                 **kwargs) -> None:

        super().__init__()
        num_genre = kwargs['num_genres']
        self.input_feats = input_feats 
        self.latent_dim = latent_dim
        self.code_dim_t = code_dim 
        self.code_dim_b = code_dim * 2
        self.num_tokens = num_tokens
        self.dropout = dropout
        self.activation = activation
        self.norm = norm
        self.cond_drop_prob = cond_drop_prob
        self.genre_emb_dim = kwargs['genre_emb_dim']
        mfcc_dim = 40
        chroma_dim = 12
        self.mfcc_encoder = Encoder(mfcc_dim,
                               output_emb_width//4,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)
        self.chroma_encoder = Encoder(chroma_dim,
                               output_emb_width//4,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)
        self.genre_emb = nn.Linear(num_genre, self.genre_emb_dim)
        self.music_feats_enc_dim = (output_emb_width//4) + (output_emb_width//4)
        self.position_enc_ = PositionalEncoding(self.latent_dim, self.dropout, max_len=3000)
        self.mask_id = self.num_tokens
        self.token_emb = nn.Embedding(self.num_tokens+1, self.code_dim_b)
        self.cond_top_token_emb = nn.Embedding(self.num_tokens+1, self.code_dim_t)
        self.token_lin = nn.Linear(self.code_dim_b, self.latent_dim)
        self.cond_top_token_lin = nn.Linear(self.code_dim_t, self.latent_dim)
        self.cond_lin = nn.Linear(self.music_feats_enc_dim, self.latent_dim)
        
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation=self.activation,
                                                          batch_first=True)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)
        
        self.output_process = OutputProcess(out_feats=self.num_tokens, 
                                            latent_dim=latent_dim, 
                                            activation=self.activation)
        if kwargs['noise_schedule'] == 'cosine':
            self.noise_schedule = cosine_schedule
        elif kwargs['noise_schedule'] == 'scaled_cosine':
            self.noise_schedule = scale_cosine_schedule
        elif kwargs['noise_schedule'] == 'q':
            self.noise_schedule = q_schedule
        else:   self.noise_schedule = linear_schedule
       
    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def mask_cond(self, cond, force_mask=False):   
        bs, t, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, t), device=cond.device) * self.cond_drop_prob).view(bs, t, 1)
            return cond * (1. - mask)
        else:
            return cond
    
    def internal_forward(self, x_ids: Tensor, top_code_idx_cond: Tensor, mfcc_feats: Tensor, chroma_feats: Tensor, force_mask=False):
        B, token_len = x_ids.shape
        mfcc_in = self.mfcc_encoder(self.preprocess(mfcc_feats))
        chroma_in = self.chroma_encoder(self.preprocess(chroma_feats))
        x_cond = self.preprocess(torch.cat((mfcc_in, chroma_in), dim=1))
        x_cond = self.cond_lin(self.mask_cond(x_cond, force_mask=force_mask))

        top_code_idx_emb = self.cond_top_token_emb(top_code_idx_cond)
        cond_top_token_embeddings = self.cond_top_token_lin(top_code_idx_emb)
        code_idx_emb = self.token_emb(x_ids)
        token_embeddings = self.token_lin(code_idx_emb)
        seq = torch.cat((x_cond, cond_top_token_embeddings, token_embeddings), dim=1)
        trans_enc_input_seq = self.position_enc_(seq)
        trans_enc_output_seq = self.seqTransEncoder(trans_enc_input_seq)
        logits = self.output_process(trans_enc_output_seq)
        return logits[..., -token_len:]
    
    def forward(self, code_idx_combined: Tensor, mfcc_feats: Tensor, chroma_feats: Tensor, iteration, bert_masking_scheme=True):
        code_idx = code_idx_combined[0]
        code_idx_cond = code_idx_combined[1]
        B, num_tokens = code_idx.shape
        x_ids = code_idx.clone()
        # r2 = iteration/45000 if iteration/45000 < 1.0 else 1.0
        rand_time = uniform((B,), device=x_ids.device, r1=0, r2=1.0)  #random hard-code
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (num_tokens * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((B, num_tokens), device=x_ids.device).argsort(dim=-1)
        mask_token = batch_randperm < num_token_masked.unsqueeze(-1)
        masked_labels = torch.where(mask_token, code_idx, self.mask_id)

        if bert_masking_scheme and iteration > 5000:
            # Further Apply Bert Masking Scheme
            # Step 1: 10% replace with an incorrect token
            mask_rid = get_mask_subset_prob(mask_token, 0.1)
            rand_id = torch.randint_like(x_ids, high=self.num_tokens)
            x_ids = torch.where(mask_rid, rand_id, x_ids)
            # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
            mask_mid = get_mask_subset_prob(mask_token & ~mask_rid, 0.88)
            x_ids = torch.where(mask_mid, self.mask_id, x_ids)        
        else:
            x_ids = torch.where(mask_token, self.mask_id, x_ids)
            
        logits = self.internal_forward(x_ids, code_idx_cond, mfcc_feats, chroma_feats, force_mask=False)
        
        masked_ce_loss, pred_idx, acc = calc_performance(logits, masked_labels, ignore_index=self.mask_id) 

        # Trying this out because validation loss doesn't decrease
        # ce_weights = torch.ones(self.num_tokens).to(code_idx.device) * 0.5
        # mean_proximity_loss = weightedSequenceCrossEntropyLoss(logits, code_idx, ce_weights, window_size=7)
            
        return pred_idx, masked_ce_loss, acc
    
    def forward_with_cond_scale(self, x_ids: Tensor, code_idx_cond: Tensor, mfcc_feats: Tensor, chroma_feats: Tensor,  cond_scale=3, force_mask=False):
        # bs = motion_ids.shape[0]
        # if cond_scale == 1:
        if force_mask:
            return self.internal_forward(x_ids, code_idx_cond, mfcc_feats, chroma_feats, force_mask=True)

        logits = self.internal_forward(x_ids, code_idx_cond, mfcc_feats, chroma_feats)
        if cond_scale == 1:
            return logits

        aux_logits = self.internal_forward(x_ids, code_idx_cond, mfcc_feats, chroma_feats, force_mask=True)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self, code_idx_cond: Tensor, mfcc_feats: Tensor, chroma_feats: Tensor, token_len, timesteps=20):
        
        starting_temperature = 1
        topk_filter_thres = 0.9
        cond_scale = 3
        gsample = True
        m_lens = torch.tensor([token_len]).to(mfcc_feats.device)
        padding_mask = ~lengths_to_mask(m_lens, token_len)
        # Start from all tokens being masked
        x_ids = torch.where(padding_mask, self.mask_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.)
        
        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=mfcc_feats.device), reversed(range(timesteps))):
            # 0 < timestep < 1
            # rand_time = uniform((B,), device=x_cond.device, r1=0, r2=1.0)  # Tensor
            rand_mask_prob = self.noise_schedule(timestep)
            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = (token_len * rand_mask_prob).round().clamp(min=1)

            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            x_ids = torch.where(is_mask, self.mask_id, x_ids)

            logits = self.forward_with_cond_scale(x_ids, code_idx_cond, mfcc_feats, chroma_feats, cond_scale=cond_scale)
            logits = logits.permute(0, 2, 1)   # (b, seqlen, ntoken)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            # else:
            # temperature = starting_temperature * (steps_until_x0 / timesteps)
            # temperature = max(temperature, 1e-4)
            # print(filtered_logits.shape)
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                
                probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)

            # print(pred_ids.max(), pred_ids.min())
            # if pred_ids.
            x_ids = torch.where(is_mask, pred_ids, x_ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        x_ids = torch.where(padding_mask, -1, x_ids)
        # print("Final", ids.max(), ids.min())
        return x_ids

    def generate_edit(self, code_idx_cond: Tensor, mfcc_feats: Tensor, chroma_feats: Tensor, token_len, timesteps=20):
        
        starting_temperature = 1
        topk_filter_thres = 0.9
        cond_scale = 3
        gsample = True
        m_lens = torch.tensor([token_len]).to(mfcc_feats.device)
        padding_mask = ~lengths_to_mask(m_lens, token_len)
        edit_mask = padding_mask.clone()
        edit_mask[:, 0:10] = True
        edit_mask[:, -10:] = True
        x_ids = torch.where(edit_mask, self.mask_id, self.mask_id)
        scores = torch.where(edit_mask, 1e5, 0.)
        
        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=mfcc_feats.device), reversed(range(timesteps))):
            # 0 < timestep < 1
            # rand_time = uniform((B,), device=x_cond.device, r1=0, r2=1.0)  # Tensor
            rand_mask_prob = self.noise_schedule(timestep)
            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = (token_len * rand_mask_prob).round().clamp(min=1)

            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            x_ids = torch.where(is_mask, self.mask_id, x_ids)

            logits = self.forward_with_cond_scale(x_ids, code_idx_cond, mfcc_feats, chroma_feats, cond_scale=cond_scale)
            logits = logits.permute(0, 2, 1)   # (b, seqlen, ntoken)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            # else:
            # temperature = starting_temperature * (steps_until_x0 / timesteps)
            # temperature = max(temperature, 1e-4)
            # print(filtered_logits.shape)
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                
                probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)

            # print(pred_ids.max(), pred_ids.min())
            # if pred_ids.
            x_ids = torch.where(is_mask, pred_ids, x_ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        # x_ids = torch.where(padding_mask, -1, x_ids)
        # print("Final", ids.max(), ids.min())
        return x_ids

