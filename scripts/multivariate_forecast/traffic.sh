export CUDA_VISIBLE_DEVICES=0

model_name=DSIN_PMA

for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path traffic.csv \
      --model_id traffic_96_96_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers_time 1 \
      --e_layers_variate 4 \
      --enc_in 862 \
      --des 'Exp' \
      --d_model 256 \
      --batch_size 4 \
      --dropout 0.1 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --n_heads 16 \
      --alpha 0.4 
done
