#export CUDA_VISIBLE_DEVICES=1

model_name=DSIN_PMA

for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path exchange_rate.csv \
      --model_id exchange_rate_96_96_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers_time 1 \
      --e_layers_variate 1 \
      --factor 3 \
      --enc_in 8 \
      --des 'Exp' \
      --d_model 256 \
      --batch_size 32 \
      --dropout 0.3 \
      --learning_rate 0.0001 \
      --train_epochs 3 \
      --n_heads 4
done


