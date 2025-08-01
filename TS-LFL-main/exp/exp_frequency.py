import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
from models.FreForecaster import FreModel
import copy
warnings.filterwarnings('ignore')


class Exp_Frequency(Exp_Basic):
    def __init__(self, args, ft=False, path=False):
        self.ft = ft
        self.avg_speed = None
        self.path = path
        self.decay = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2]
        self.patch_len = 16
        self.mask = args.mask
        self.pred_len = args.pred_len
        self.pattern = args.pattern
        super(Exp_Frequency, self).__init__(args, ft, path)

    def _build_model(self):

        model = FreModel(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()
        
        f_dim = 0

        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        outputs = outputs[:, :, f_dim:]

        return outputs, batch_y

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
                
                spec_1 = torch.fft.rfft(batch_y, dim=1)
                spec_1[:,-int(self.mask * (self.pred_len / 2 + 1)):,:] = 0
                batch_y = torch.fft.irfft(spec_1, dim=1)

                batch_y = batch_y.permute(0, 2, 1).unfold(dimension=-1, size=self.patch_len, step=self.patch_len)     # x: [B, L, C] -> [B, C, L] -> [B, C, N, P]
                batch_y = torch.fft.rfft(batch_y, dim=-1, norm='forward')                        # x: [B, C, N, P] -> [B, C, N, P // 2 + 1]
                batch_y = batch_y.reshape(batch_y.shape[0], batch_y.shape[1], -1, 1).squeeze(-1)       

                loss = torch.sum(torch.sum(torch.sum(torch.abs(outputs - batch_y)))) / (outputs.shape[1] * outputs.shape[0])

                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        speed_list = []
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            iter_tol = 0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                iter_tol += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                # Low-pass filtering
                spec_1 = torch.fft.rfft(batch_y, dim=1)
                spec_1[:,-int(self.mask * (self.pred_len / 2 + 1)):,:] = 0
                batch_y = torch.fft.irfft(spec_1, dim=1)

                # Non-overlapping Patching
                batch_y = batch_y.permute(0, 2, 1).unfold(dimension=-1, size=self.patch_len, step=self.patch_len) 
                batch_y = torch.fft.rfft(batch_y, dim=-1, norm='forward')  
                batch_y = batch_y.reshape(batch_y.shape[0], batch_y.shape[1], -1, 1).squeeze(-1) 
                
                # FALoss
                loss = torch.sum(torch.sum(torch.sum(torch.abs(outputs - batch_y)))) / (outputs.shape[1] * outputs.shape[0])

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            total_time = time.time() - epoch_time
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print("Average Speed: {}".format(total_time / iter_tol))
            speed_list.append(total_time / iter_tol)
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))
        self.avg_speed = round(np.mean(speed_list) * 1000, 2)
        return best_model_path
