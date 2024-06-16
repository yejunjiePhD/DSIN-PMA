#export CUDA_VISIBLE_DEVICES=1

model_name=DSIN_PMA


for pred_len in 96
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_96_96_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers_time 1 \
      --e_layers_variate 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 128 \
      --d_ff 128 \
      --batch_size 32 \
      --dropout 0.1 \
      --learning_rate 0.0005 \
      --train_epochs 20 \
      --patience 3 \
      --n_heads 8 \
      --alpha 0.4
done


for pred_len in 192 336
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_96_96_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers_time 1 \
      --e_layers_variate 2 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 128 \
      --d_ff 128 \
      --batch_size 16 \
      --dropout 0.2 \
      --learning_rate 0.0005 \
      --train_epochs 20 \
      --patience 5 \
      --n_heads 8 \
      --alpha 0.4
done

for pred_len in 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_96_96_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers_time 1 \
      --e_layers_variate 2 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 128 \
      --d_ff 128 \
      --batch_size 32 \
      --dropout 0.1 \
      --learning_rate 0.0001 \
      --train_epochs 20 \
      --patience 3 \
      --n_heads 8 \
      --alpha 0
done

