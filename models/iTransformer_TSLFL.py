
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.patch_len = configs.TSLFL_patch_len
        self.pre_patch_len = configs.out_patch_len
        self.main_fre = configs.main_fre
        self.patch_num = int(self.pred_len / self.patch_len)


    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def TSLFL_forcast(self, x_enc, spec_in):
        # # Encoder
        B,T,N = x_enc.shape
        spec_x = torch.fft.rfft(x_enc, dim=1, norm='forward')
        spec_x[:, -int(self.seq_len / 2):, :] = 0
        low_x = torch.fft.irfft(spec_x, dim=1, norm='forward')
        x_enc = x_enc - low_x   # high frequency component

        dec_out = self.forecast(x_enc)
        pred = dec_out[:, -self.pred_len:, :]

        # spec_in = spec_in.reshape(spec_in.shape[0], spec_in.shape[1], int(self.pred_len / self.pre_patch_len), -1)
        # spec_temp: [B, C, N, Pf] -> [B, C, N, P]
        spec_temp = torch.fft.irfft(spec_in, dim=-1, norm='forward', n=self.pred_len)
        # spec_temp: [B, C, N, P] -> [B, C, L]
        spec_temp = spec_temp.reshape(spec_temp.shape[0], spec_temp.shape[1], -1)
        # spec_temp = torch.einsum('bcl,ll->bcl', spec_temp, self.outpu t_weight) + self.output_bias
        spec_temp = spec_temp.permute(0, 2, 1)

        # pred = pred + spec_temp
        high_freq_pred = pred
        low_freq_pred = spec_temp

        return low_freq_pred, high_freq_pred


    def forward(self, x_enc, spec_in, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.TSLFL_forcast(x_enc, spec_in)
