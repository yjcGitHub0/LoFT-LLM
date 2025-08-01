export CUDA_VISIBLE_DEVICES=6

# model_name=DLinear_TSLFL
model_name=iTransformer_TSLFL

# python -u run_plfm.py \
#   --task_name preModel \
#   --is_training 1 \
#   --root_path ./dataset/ProductAmt/ \
#   --data_path ProductAmt.csv \
#   --model_id ProductAmt_30_6 \
#   --model PLFM \
#   --data 100_100 \
#   --features M \
#   --seq_len 30 \
#   --label_len 0 \
#   --pred_len 6 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 88 \
#   --dec_in 88 \
#   --c_out 88 \
#   --des 'Exp' \
#   --itr 1 \
#   --out_patch_len 2

python -u run_TSLFL_LLM_data.py \
  --task_name TSLFL_Forcast \
  --is_training 1 \
  --root_path ./dataset/ProductAmt/ \
  --data_path ProductAmt_with_info.csv \
  --model_id ProductAmt_30_6 \
  --model $model_name \
  --data financial_llm_train \
  --features M \
  --seq_len 30 \
  --label_len 0 \
  --pred_len 6 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 88 \
  --dec_in 88 \
  --c_out 88 \
  --des 'Exp' \
  --itr 1 \
  --out_patch_len 8 \
  --TSLFL_patch_len 8