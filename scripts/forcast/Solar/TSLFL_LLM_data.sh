export CUDA_VISIBLE_DEVICES=6

# model_name=DLinear_TSLFL
model_name=iTransformer_TSLFL

python -u run_TSLFL_LLM_data.py \
  --task_name Solar_train_data \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path Solar_Info.csv \
  --model_id Solar_data \
  --model $model_name \
  --data solar_llm_train \
  --features M \
  --seq_len 72 \
  --label_len 0 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --out_patch_len 16 \
  --TSLFL_patch_len 16