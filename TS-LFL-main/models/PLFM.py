import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FreModel(nn.Module):
    def __init__(self, configs):
        super(FreModel, self).__init__()
        self.hid_dim = 512
        self.patch_len = 16
        self.out_patch_len = 16
        self.stride = 4

        self.pred_len = configs.pred_len
        self.feature_size = configs.enc_in #channels
        self.seq_length = configs.seq_len
        
        self.patch_num = int((self.seq_length - self.patch_len) / self.stride + 1)
        self.input_len = int(self.patch_num * (self.patch_len / 2 + 1))
        self.output_len = int((self.pred_len / self.out_patch_len) * (self.out_patch_len / 2 + 1))

        self.real_linear1 = nn.Linear(self.input_len, self.hid_dim)
        self.real_linear2 = nn.Linear(self.hid_dim, self.output_len)
        self.imag_linear1 = nn.Linear(self.input_len, self.hid_dim)
        self.imag_linear2 = nn.Linear(self.hid_dim, self.output_len)

        self.act = nn.Identity()
        self.dropout = nn.Dropout(0.3)


    # frequency temporal learner
    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, C = x.shape
        x = x.permute(0, 2, 1)
        
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)     # x: [B, L, C] -> [B, C, L] -> [B, C, N, P]
        x = torch.fft.rfft(x, dim=-1, norm='forward')
        x = x.reshape(B, C, -1)

        real_x, imag_x = x.real, x.imag

        real_x = self.act(self.real_linear1(real_x))
        real_x = self.real_linear2(self.dropout(real_x))

        imag_x = self.act(self.imag_linear1(imag_x))
        imag_x = self.imag_linear2(self.dropout(imag_x))

        y = torch.stack([real_x, imag_x], dim=-1)
 
        y = torch.view_as_complex(y)
        
        return y

