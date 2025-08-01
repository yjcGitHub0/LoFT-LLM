export CUDA_VISIBLE_DEVICES=0

# model_name=iTransformer
# model_name=FITS
# model_name=TimesNet
model_name=PatchTST
# model_name=DLinear
# model_name=FreTS
# model_name=TimeXer
# model_name=Transformer

# plens=(1 2 3 4 5 6 7 8 9 10)
plens=(1)
seeds=(1)

for pl in ${plens[@]}; do
  for seed in ${seeds[@]}; do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/Solar/ \
      --data_path Solar.csv \
      --model_id Solar_72_x \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 72 \
      --label_len 0 \
      --pred_len $pl \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --itr 1 \
      --d_model 256 \
      --d_ff 512 \
      --seed $seed
  done
done