import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2311.06184.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.embed_size = 128  # embed_size
        self.hidden_size = 256  # hidden_size
        self.pred_len = configs.pred_len
        self.feature_size = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forecast(self, x_enc):
        # x: [Batch, Input length, Channel]
        B, T, N = x_enc.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_enc)
        bias = x
        # [B, N, T, D]
        if self.channel_independence == '0':
            x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x
    
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

    def forward(self, x_enc, spec_in, x_dec, x_mark_dec, mask=None):
        return self.TSLFL_forcast(x_enc, spec_in)
