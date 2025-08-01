export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# model_name=DLinear_TSLFL
# model_name=iTransformer_TSLFL
model_name=DLinear
plens=(1)
seeds=(1)

for seed in ${seeds[@]}; do
for pl in ${plens[@]}; do
  # python -u run.py \
  #     --task_name long_term_forecast \
  #     --is_training 1 \
  #     --root_path ./dataset/ProductAmt/ \
  #     --data_path ProductAmt.csv \
  #     --model_id ProductAmt_30_$pl \
  #     --model $model_name \
  #     --data custom \
  #     --features M \
  #     --seq_len 30 \
  #     --label_len 0 \
  #     --pred_len $pl \
  #     --e_layers 2 \
  #     --d_layers 1 \
  #     --factor 3 \
  #     --enc_in 88 \
  #     --dec_in 88 \
  #     --c_out 88 \
  #     --des 'Exp' \
  #     --itr 1 \
  #     --d_model 512 \
  #     --d_ff 1024 \
  #     --seed $seed

  python -u run_TSLFL.py \
    --task_name LLM_Ablation \
    --is_training 0 \
    --root_path ./dataset/ProductAmt/ \
    --data_path ProductAmt_with_info.csv \
    --model_id ProductAmt_30_x \
    --model $model_name \
    --data financial_info \
    --features M \
    --seq_len 30 \
    --label_len 0 \
    --pred_len $pl \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 88 \
    --dec_in 88 \
    --c_out 88 \
    --des 'Exp' \
    --itr 1 \
    --out_patch_len 8 \
    --TSLFL_patch_len 8 \
    --seed $seed

done
done