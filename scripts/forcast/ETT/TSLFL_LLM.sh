export CUDA_VISIBLE_DEVICES=2,4,6

# model_name=DLinear_TSLFL
model_name=iTransformer_TSLFL
plens=(4 5 6 7 8)
seeds=(1 2 3)

for seed in ${seeds[@]}; do
for pl in ${plens[@]}; do
  # python -u run_plfm.py \
  #   --task_name preModel \
  #   --is_training 1 \
  #   --root_path ./dataset/ProductAmt/ \
  #   --data_path ProductAmt.csv \
  #   --model_id ProductAmt_30_x \
  #   --model PLFM \
  #   --data custom \
  #   --features M \
  #   --seq_len 30 \
  #   --label_len 0 \
  #   --pred_len $pl \
  #   --e_layers 2 \
  #   --d_layers 1 \
  #   --factor 3 \
  #   --enc_in 88 \
  #   --dec_in 88 \
  #   --c_out 88 \
  #   --des 'Exp' \
  #   --itr 1 \
  #   --out_patch_len 8

  # python -u run_TSLFL.py \
  #   --task_name TSLFL_Forcast \
  #   --is_training 1 \
  #   --root_path ./dataset/ProductAmt/ \
  #   --data_path ProductAmt.csv \
  #   --model_id ProductAmt_30_x \
  #   --model $model_name \
  #   --data custom \
  #   --features M \
  #   --seq_len 30 \
  #   --label_len 0 \
  #   --pred_len $pl \
  #   --e_layers 2 \
  #   --d_layers 1 \
  #   --factor 3 \
  #   --enc_in 88 \
  #   --dec_in 88 \
  #   --c_out 88 \
  #   --des 'Exp' \
  #   --itr 1 \
  #   --out_patch_len 8 \
  #   --TSLFL_patch_len 8

  python -u run_TSLFL.py \
    --task_name LLM_ETT \
    --is_training 0 \
    --root_path ./dataset/ETT/ \
    --data_path ETTh1_Info.csv \
    --model_id ETT_96_$pl \
    --model $model_name \
    --data solar_info \
    --features M \
    --seq_len 96 \
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
    --out_patch_len 16 \
    --TSLFL_patch_len 16 \
    --seed $seed \
    --batch_size 32

done
done