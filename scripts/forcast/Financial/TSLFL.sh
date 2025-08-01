export CUDA_VISIBLE_DEVICES=4

# model_name=DLinear_TSLFL
model_name=iTransformer_TSLFL
plens=(1)
seeds=(1)

for seed in ${seeds[@]}; do
  for pl in ${plens[@]}; do
    python -u run_plfm.py \
      --task_name preModel \
      --is_training 1 \
      --root_path ./dataset/ProductAmt/ \
      --data_path ProductAmt.csv \
      --model_id ProductAmt_30_$pl \
      --model PLFM \
      --data custom \
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
      --seed $seed \
      --patience 10 \
      --train_epochs 20

    python -u run_TSLFL.py \
      --task_name TSLFL_Forcast \
      --is_training 1 \
      --root_path ./dataset/ProductAmt/ \
      --data_path ProductAmt.csv \
      --model_id ProductAmt_30_$pl \
      --model $model_name \
      --data custom \
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
      --seed $seed \
      --patience 10 \
      --train_epochs 20 \
      --learning_rate 0.001
  done
done
