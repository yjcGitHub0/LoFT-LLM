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
import copy
from models.PLFM import FreModel
import layers.FinancialLLMProcess_utils as LLMutils
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

warnings.filterwarnings('ignore')


class Exp_TSLFL_LLM(Exp_Basic):
    def __init__(self, args, ft=False, path=False):
        self.ft = ft
        self.avg_speed = None
        self.path = path
        self.decay = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2]
        self.pred_len = args.pred_len
        self.fremodel = FreModel(args).to('cuda:0')
        self.fremodel.load_state_dict(torch.load(self.path, map_location='cuda:0'), strict=False)

        # 原始基础模型路径
        base_model_name = "Qwen3/Qwen3-8B"
        # 使用QLoRA微调后保存的路径
        peft_model_path = "Qwen3/Financial_Finetune/v2-20250711-171550/checkpoint-743"
        # 加载Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side='left')
        # 加载QLoRA模型
        self.qwen_model = AutoPeftModelForCausalLM.from_pretrained(
            peft_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        super(Exp_TSLFL_LLM, self).__init__(args)

    def _build_model(self):

        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        self.feature_names = data_set.cols
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _predict(self, batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time, spec_x):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, spec_x, None, dec_inp, None)

            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()
        
        low_freq_pred, high_freq_pred = outputs
        pred = low_freq_pred + high_freq_pred

        # LLM 处理过程
        x_combin = torch.cat((batch_x, batch_x_semantic), dim=2)
        y_low_combin = torch.cat((low_freq_pred, batch_y_semantic), dim=2)
        y_high_combin = torch.cat((high_freq_pred, batch_y_semantic), dim=2)
        y_combin = torch.cat((pred, batch_y_semantic), dim=2)
        grouped_x, all_group_feature_names = LLMutils.group_features_by_suffix_gpu(x_combin, self.feature_names)
        grouped_y, _ = LLMutils.group_features_by_suffix_gpu(y_combin, self.feature_names)
        grouped_low_y, _ = LLMutils.group_features_by_suffix_gpu(y_low_combin, self.feature_names)
        grouped_high_y, _ = LLMutils.group_features_by_suffix_gpu(y_high_combin, self.feature_names)
        
        field_x = LLMutils.split_group_tensor_by_feature_name(grouped_x, all_group_feature_names)
        field_y = LLMutils.split_group_tensor_by_feature_name(grouped_y, all_group_feature_names)
        field_low_y = LLMutils.split_group_tensor_by_feature_name(grouped_low_y, all_group_feature_names)
        field_high_y = LLMutils.split_group_tensor_by_feature_name(grouped_high_y, all_group_feature_names)

        prompts = LLMutils.make_prompts_from_fields(field_x, field_y, batch_x_time, batch_y_time, self.args.pred_len,
                                        field_high_y, field_low_y)
        print(prompts[0])
        print(None.shape)

        print(f"Generated {len(prompts)} prompts for LLM processing.")

        apply_amt_refined, redeem_amt_refined = LLMutils.Qwen(self.tokenizer, self.qwen_model, prompts, self.args.pred_len,
                          field_y['apply_amt'], field_y['redeem_amt'])
        outputs_refined = LLMutils.regroup_predictions_filtered(apply_amt_refined, redeem_amt_refined, field_x['fund_id'], self.feature_names, self.args.pred_len)
        outputs_refined = outputs_refined.float().to(self.device)

        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
        # outputs = outputs[:, -self.args.pred_len:, :]

        return outputs_refined, batch_y, pred


    def test(self, setting, test=1):
        _, test_loader = self._get_data(flag='test')
        if test:
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                "TSLFL_Forcast",
                self.args.model_id,
                self.args.model,
                self.args.features,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.factor,
                self.args.embed,
                self.args.distil,
                self.args.des, 0)
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        preds = []
        trues = []
        orgs = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.fremodel.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_semantic = batch_x_semantic.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_semantic = batch_y_semantic.float().to(self.device)
                temp_batch_x = copy.deepcopy(batch_x)
                
                '''
                    Note that during the inference process, PLFM directly outputs the forecasting spectra, without test data leakage
                '''
                spec_in = self.fremodel(temp_batch_x, batch_y, None, None)

                outputs, batch_y, org = self._predict(batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time, spec_in)

                batch_y = batch_y.detach().cpu().numpy()

                outputs = outputs.detach().cpu().numpy()
                # spec_temp = spec_temp.detach().cpu().numpy()
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                org = org.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                orgs.append(org)

                preds_now = np.concatenate(preds, axis=0)
                trues_now = np.concatenate(trues, axis=0)
                orgs_now = np.concatenate(orgs, axis=0)
                mae, mse, rmse, mape, mspe, smape = metric(preds_now, trues_now)
                mae_org, mse_org, _, _, _, _ = metric(orgs_now, trues_now)
                print('mse:{}, mae:{}, mape:{}'.format(mse, mae, mape))
                print('mse_org:{}, mae_org:{}'.format(mse_org, mae_org))

        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, smape = metric(preds, trues)

        print('mse:{}, mae:{} bs:{} lr:{} dm:{} df:{} main_fre:{}'.format(mse, mae, self.args.batch_size, self.args.learning_rate, self.args.d_model, self.args.d_ff, self.args.main_fre))
        # print('mse:{}, mae:{}'.format(mse, mae))

        f = open("LLM_results.txt", 'a') 
        f.write(setting + "  \n")
        f.write(f'seed: {self.args.seed}, pred_len: {self.pred_len}\n')
        f.write('mae, mse, mape:\n')
        f.write(f'{mae}\t{mse}\t{mape}')
        # f.write('mse:{}, mae:{} bs:{} lr:{} dm:{} df:{}  main_fre:{}'.format(mse, mae, self.args.batch_size, self.args.learning_rate, self.args.d_model, self.args.d_ff, self.args.main_fre))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            logging.info(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                spec_x = self.fremodel(batch_x, batch_y, batch_x_mark, batch_y_mark)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark, spec_x)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
