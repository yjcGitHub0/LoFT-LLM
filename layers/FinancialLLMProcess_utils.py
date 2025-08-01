import torch
import numpy as np
from collections import defaultdict
import re
from typing import List, Dict
from layers.GetFinancialPrompt import get_prompt
import math
import ast
from tqdm import tqdm

def group_features_by_suffix_gpu(tensor: torch.Tensor, feature_names: np.ndarray):
    """
    Args:
        tensor: Tensor of shape [B, L, C], expected on CPU or GPU.
        feature_names: numpy array of shape [C], each element like 'apply_amt_67', 'yield', etc.

    Returns:
        grouped_tensor: Tensor of shape [B * num_groups, group_len, L] (on GPU)
        group_keys: List of group ids (e.g. ['62', '67'])
        group_features: Dict[group_id] -> List of feature names
        all_group_feature_names_np: 2D numpy array where each sublist contains feature names for a column in grouped_tensor
    """
    B, L, C = tensor.shape

    device = tensor.device

    # Step 1: Assign features to groups and collect global features
    group_to_indices = defaultdict(list)
    global_indices = []
    global_features = []

    for idx, name in enumerate(feature_names):
        match = re.search(r'_(\d+)$', name)
        if match:
            group_id = match.group(1)  # e.g., '67'
            group_to_indices[group_id].append(idx)
        else:
            global_indices.append(idx)
            global_features.append(name)

    # Initialize group_features dictionary
    group_features = {}

    # Add global features to each group
    for group_id in group_to_indices:
        group_to_indices[group_id].extend(global_indices)
        group_features[group_id] = [feature_names[i] for i in group_to_indices[group_id]]

    # Step 2: Group features by id
    # group_keys = sorted(group_to_indices.keys())
    group_keys = group_to_indices.keys()
    group_tensors = []
    all_group_feature_names = []

    for group_id in group_keys:
        indices = group_to_indices[group_id]
        selected = tensor[:, :, indices]  # [B, L, group_len]
        group_tensors.append(selected)    # List of [B, L, ?]
        feature_list = [feature_names[i] for i in indices]
        all_group_feature_names.append([feature_list] * B)

    # Convert all_group_feature_names to a 3D numpy array of objects
    all_group_feature_names_np = np.array(all_group_feature_names, dtype=object)

    # Transpose all_group_feature_names_np to match the order of grouped_tensor
    all_group_feature_names_np = all_group_feature_names_np.transpose(1, 0, 2)

    # Flatten the first two dimensions of all_group_feature_names_np
    all_group_feature_names_np = all_group_feature_names_np.reshape(-1, all_group_feature_names_np.shape[2])

    # Step 3: Stack → [B, L, num_groups, group_len]
    stacked = torch.stack(group_tensors, dim=2)

    # Step 4: Permute to → [B, num_groups, group_len, L]
    grouped_tensor = stacked.permute(0, 2, 3, 1).to(device)

    # Reshape grouped_tensor to merge the first two dimensions
    grouped_tensor = grouped_tensor.reshape(B * len(group_keys), -1, L)

    return grouped_tensor, all_group_feature_names_np


def split_group_tensor_by_feature_name(
    group_tensor, group_feature_names
):

    field_data = {
        'fund_id': None,
        'apply_amt': None,
        'redeem_amt': None,
        'yield_': None,
        'uv_fundown': None,
        'uv_stableown': None,
        'uv_fundopt': None,
        'uv_fundmarket': None,
        'uv_termmarket': None,
        'is_trade': None,
        'is_month_end': None,
    }

    # Extract fund_id from the feature names
    fund_ids = []
    for i in range(group_feature_names.shape[0]):
        for name in group_feature_names[i]:
            match = re.search(r'_(\d+)$', name)
            if match:
                fund_ids.append(match.group(1))
                break

    # Assign fund_id to field_data
    field_data['fund_id'] = fund_ids

    for i, name in enumerate(group_feature_names[1]):
        for key in field_data.keys():
            if name.startswith(key):
                field_data[key] = group_tensor[:, i, :]
                break

    return field_data


def make_prompts_from_fields(field_x, field_y, time_x, time_y, pred_len, 
                             filed_y_high_freq, filed_y_low_freq):
    """
    输入:
        field_x: dict, 历史区间各字段，shape: [B, T]
        field_y: dict, 预测区间各字段，shape: [B, P]
    输出:
        prompts: List[str], 每个batch一个prompt
    """
    # 取batch数
    B = len(field_x['fund_id'])
    # 扩展时间为 [B*44, T]
    time_x = np.repeat(np.array(time_x)[:, np.newaxis, :], 44, axis=1).reshape(B, -1)
    time_y = np.repeat(np.array(time_y)[:, np.newaxis, :], 44, axis=1).reshape(B, -1)
    # time_x = time_x.unsqueeze(1).expand(-1, 44, -1).reshape(B, -1)
    # time_y = time_y.unsqueeze(1).expand(-1, 44, -1).reshape(B, -1)
    
    prompts = []
    for b in range(B):
        prompt = get_prompt(
            fund_id=field_x['fund_id'][b],
            dates=time_x[b],
            apply_amt=field_x['apply_amt'][b],
            redeem_amt=field_x['redeem_amt'][b],
            yield_=field_x['yield_'][b],
            uv_fundown=field_x['uv_fundown'][b],
            uv_stableown=field_x['uv_stableown'][b],
            uv_fundopt=field_x['uv_fundopt'][b],
            uv_fundmarket=field_x['uv_fundmarket'][b],
            is_month_end=field_x['is_month_end'][b],
            is_trade=field_x['is_trade'][b],
            pred_dates=time_y[b],
            predicted_apply=field_y['apply_amt'][b],
            predicted_redeem=field_y['redeem_amt'][b],
            predicted_apply_high_freq=filed_y_high_freq['apply_amt'][b],
            predicted_apply_low_freq=filed_y_low_freq['apply_amt'][b],
            predicted_redeem_high_freq=filed_y_high_freq['redeem_amt'][b],
            predicted_redeem_low_freq=filed_y_low_freq['redeem_amt'][b],
            pred_yield_=field_y['yield_'][b],
            pred_uv_fundown=field_y['uv_fundown'][b],
            pred_uv_stableown=field_y['uv_stableown'][b],
            pred_uv_fundopt=field_y['uv_fundopt'][b],
            pred_uv_fundmarket=field_y['uv_fundmarket'][b],
            pred_is_month_end=field_y['is_month_end'][b],
            pred_is_trade=field_y['is_trade'][b],
            pred_len=pred_len
        )
        prompts.append(prompt)
    return prompts


def convert_string_to_float_list(string_list, pred_len):
    # 匹配所有 float 或 int（支持负号）
    numbers = re.findall(r'-?\d+(?:\.\d+)?', string_list)

    try:
        float_numbers = [float(num) for num in numbers]
        # 截断或补齐
        if len(float_numbers) != pred_len: print(string_list)
        if len(float_numbers) < pred_len:
            float_numbers += [-1000.0] * (pred_len - len(float_numbers))
        else:
            float_numbers = float_numbers[:pred_len]
        return float_numbers
    except Exception as e:
        print(f"Error converting to float: {e}")
        return [-1000.0] * pred_len



def Qwen(tokenizer, model, prompts, pred_len, apply_amt_org, redeem_amt_org, batch_size=16):
    apply_amt_refined = []
    redeem_amt_refined = []

    num_batches = math.ceil(len(prompts) / batch_size)

    for batch_idx in tqdm(range(num_batches), desc="Qwen Generating"):
        batch_prompts = prompts[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # 构建输入文本
        texts = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # 禁用“思考模式”
            )
            texts.append(text)

        # 编码并推理
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256
        )

        # 解码与解析
        for i in range(len(batch_prompts)):
            output_ids = generated_ids[i][len(model_inputs.input_ids[i]):].tolist()
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            lines = content.strip().split('\n')
            if len(lines) >= 2:
                apply_amt = convert_string_to_float_list(lines[0], pred_len)
                redeem_amt = convert_string_to_float_list(lines[1], pred_len)
            else:
                print(f"Warning: Unexpected output format for prompt {batch_idx * batch_size + i}. Using default values.")
                apply_amt = [-1000.0] * pred_len
                redeem_amt = [-1000.0] * pred_len
            
            j = batch_idx * batch_size + i

            a_org = apply_amt_org[j].cpu().numpy()
            r_org = redeem_amt_org[j].cpu().numpy()
            apply_amt_np = np.array(apply_amt)
            redeem_amt_np = np.array(redeem_amt)
            # print(content)
            for k in range(apply_amt_np.shape[0]):
                # if power_np[k] != 0 and np.abs(power_np[k] - p_org[k]) > 1e-4:
                if apply_amt_np[k] != 0 and np.abs(apply_amt_np[k] - a_org[k]) > 100:
                    apply_amt_np[k] = a_org[k]
                if redeem_amt_np[k] != 0 and np.abs(redeem_amt_np[k] - r_org[k]) > 100:
                    redeem_amt_np[k] = r_org[k] 

            # print(apply_amt_np)
            apply_amt = apply_amt_np.tolist()
            redeem_amt = redeem_amt_np.tolist()
            apply_amt_refined.append(apply_amt)
            redeem_amt_refined.append(redeem_amt)

    return apply_amt_refined, redeem_amt_refined

def regroup_predictions_filtered(apply_amt_refined, redeem_amt_refined, fund_ids, feature_names, pred_len):
    """
    输出: tensor [B, P, C2]，C2为feature_names中所有apply/redeem相关列，顺序与feature_names一致
    """
    # 统计所有apply_amt/redeem_amt相关的fund_id及其在fund_ids中的索引
    B = len(fund_ids) // len(set(fund_ids))
    group_per_batch = len(fund_ids) // B

    # 建立fund_id到group idx的映射
    fundid2groupidx = {}
    for b in range(B):
        for g in range(group_per_batch):
            idx = b * group_per_batch + g
            fundid2groupidx[(b, fund_ids[idx])] = idx

    C2 = 0
    col_types = []
    col_fundids = []
    for name in feature_names:
        if name.startswith('apply_amt_'):
            col_types.append('apply')
            col_fundids.append(name.split('_')[-1])
            C2 += 1
        elif name.startswith('redeem_amt_'):
            col_types.append('redeem')
            col_fundids.append(name.split('_')[-1])
            C2 += 1

    out_tensor = torch.zeros(B, pred_len, C2)
    for b in range(B):
        for c, (typ, fundid) in enumerate(zip(col_types, col_fundids)):
            if typ == 'apply':
                key = (b, fundid)
                if key in fundid2groupidx:
                    idx = fundid2groupidx[key]
                    out_tensor[b, :, c] = torch.tensor(apply_amt_refined[idx])
            elif typ == 'redeem':
                key = (b, fundid)
                if key in fundid2groupidx:
                    idx = fundid2groupidx[key]
                    out_tensor[b, :, c] = torch.tensor(redeem_amt_refined[idx])
            # else: 保持为0
    return out_tensor


# def regroup_predictions_filtered(apply_amt_refined, redeem_amt_refined, fund_ids, feature_names, P):
#     """
#     只保留feature_names中以apply_amt_或redeem_amt_开头的列，按fund_id顺序组合预测结果
#     返回: apply_tensor, redeem_tensor, shape [B, P, C1]
#     """
#     # 找到所有apply_amt/redeem_amt相关的列及其fund_id
#     apply_cols = []
#     redeem_cols = []
#     apply_fund_ids = []
#     redeem_fund_ids = []
#     for i, name in enumerate(feature_names):
#         if name.startswith('apply_amt_'):
#             apply_cols.append(i)
#             apply_fund_ids.append(name.split('_')[-1])
#         if name.startswith('redeem_amt_'):
#             redeem_cols.append(i)
#             redeem_fund_ids.append(name.split('_')[-1])

#     B = len(fund_ids) // len(set(fund_ids))
#     C1 = len(apply_cols)  # apply和redeem的fund_id顺序应一致
#     apply_tensor = torch.zeros(B, P, C1)
#     redeem_tensor = torch.zeros(B, P, C1)

#     group_per_batch = len(fund_ids) // B
#     for b in range(B):
#         for g in range(group_per_batch):
#             idx = b * group_per_batch + g
#             fund_id = fund_ids[idx]
#             # 找到该fund_id在apply/redeem中的列
#             if fund_id in apply_fund_ids:
#                 c = apply_fund_ids.index(fund_id)
#                 apply_tensor[b, :, c] = torch.tensor(apply_amt_refined[idx])
#             if fund_id in redeem_fund_ids:
#                 c = redeem_fund_ids.index(fund_id)
#                 redeem_tensor[b, :, c] = torch.tensor(redeem_amt_refined[idx])
#     return apply_tensor, redeem_tensor, apply_cols, redeem_cols
