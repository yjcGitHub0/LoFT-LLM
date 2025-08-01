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
from transformers import AutoTokenizer, AutoModelForCausalLM
import layers.FinancialLLMProcess_utils as LLMutils
import json
import random

warnings.filterwarnings('ignore')


class Exp_TSLFL_LLM_Train_Data(Exp_Basic):
    def __init__(self, args, ft=False, path=False):
        self.args = args
        super(Exp_TSLFL_LLM_Train_Data, self).__init__(args)

    def init(self):
        self.ft = False
        self.setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            self.args.task_name,
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
            self.args.des, '0')
        pre_setting = '{}_{}_{}_{}_{}'.format(
            "preModel",
            self.args.model_id,
            "PLFM",
            self.args.seq_len,
            self.args.pred_len,
            )
        
        self.fremodel_path = os.path.join(self.args.checkpoints, pre_setting) + "/checkpoint.pth"
        self.pred_len = self.args.pred_len
        self.fremodel = FreModel(self.args).to('cuda:0')
        self.fremodel.load_state_dict(torch.load(self.fremodel_path, map_location='cuda:0'), strict=False)

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)



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


    def tensors_to_string_list(self, tensor1, tensor2):
        tensor1 = tensor1.detach().cpu().numpy()
        tensor2 = tensor2.detach().cpu().numpy()

        B, T = tensor1.shape
        string_list = []

        for i in range(B):
            # 将第一个张量的第i行转换为字符串
            str1 = ', '.join(f"{x:.3f}" for x in tensor1[i])
            str1 = f"[{str1}]"

            # 将第二个张量的第i行转换为字符串
            str2 = ', '.join(f"{x:.3f}" for x in tensor2[i])
            str2 = f"[{str2}]"

            # 合并两个字符串并添加到列表中
            combined_str = f"{str1}\n{str2}"
            string_list.append(combined_str)

        return string_list


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
        # 处理batch_y
        trues_combin = torch.cat((batch_y, batch_y_semantic), dim=2)
        grouped_trues, _ = LLMutils.group_features_by_suffix_gpu(trues_combin, self.feature_names)
        field_trues = LLMutils.split_group_tensor_by_feature_name(grouped_trues, all_group_feature_names)

        trues = self.tensors_to_string_list(field_trues['apply_amt'], field_trues['redeem_amt'])

        return prompts, trues
    

    def ndarray_to_string(self, arr):
        # 使用numpy.array2string并将参数设置为去掉空格和括号，并保留三位小数
        arr_str = np.array2string(arr, formatter={'float_kind': lambda x: f"{x:.3f}"}, separator=', ', max_line_width=np.inf)
        # 去掉两边的方括号
        arr_str = arr_str.strip('[]')
        return f"[{arr_str}]"
    
    
    def generate_json_data(self, query_list, response_list, output_json_file_path):
        data_list = []

        # 确保两个列表长度相同
        if len(query_list) != len(response_list):
            raise ValueError("查询列表和响应列表的长度必须相同")

        for query, response in zip(query_list, response_list):
            message_pair = {
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response}
                ]
            }
            data_list.append(message_pair)

        with open(output_json_file_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(data_list, jsonfile, ensure_ascii=False, indent=4)


    def test(self, setting, test=0):
        checkpoints = [
            "TSLFL_Forcast_ProductAmt_30_9_iTransformer_TSLFL_ftM_sl30_ll0_pl9_dm512_nh8_el2_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0",
            "TSLFL_Forcast_ProductAmt_30_10_iTransformer_TSLFL_ftM_sl30_ll0_pl10_dm512_nh8_el2_dl1_df512_fc3_ebtimeF_dtTrue_Exp_0",]
        
        prompts_list = []
        trues_list = []
        self.args.pred_len = 8
        for path in checkpoints:
            self.args.pred_len += 1
            self.args.model_id = 'ProductAmt_' + str(self.args.seq_len) + '_' + str(self.args.pred_len)
            self.init()
            _, test_loader = self._get_data(flag='train')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + path, 'checkpoint.pth'), map_location='cuda:0'))

            self.model.eval()
            self.fremodel.eval()
            with torch.no_grad():
                for i, (batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_x_semantic = batch_x_semantic.float().to(self.device)
                    # batch_x_time = batch_x_time.int().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_y_semantic = batch_y_semantic.float().to(self.device)
                    # batch_y_time = batch_y_time.int().to(self.device)
                    temp_batch_x = copy.deepcopy(batch_x)
                    
                    '''
                        Note that during the inference process, PLFM directly outputs the forecasting spectra, without test data leakage
                    '''
                    spec_in = self.fremodel(temp_batch_x, batch_y, None, None)

                    prompts, trues = self._predict(batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time, spec_in)
                    for j in range(len(prompts)):
                        if random.random() < 0.4:
                            prompts_list.append(prompts[j])
                            trues_list.append(trues[j])
                    
                    print(f"Processed batch {i + 1}/{len(test_loader)}")
                    print(len(prompts_list), len(trues_list))

        
        self.generate_json_data(prompts_list, trues_list, "Financial_LLM_train_data.json")



        return
