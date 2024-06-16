export CUDA_VISIBLE_DEVICES=1


model_name=DSIN_PMA

for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/Multi-domain/ \
      --data_path Multi-domain-3.csv \
      --model_id Multi_domain_3_96_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers_time 1 \
      --e_layers_variate 1 \
      --factor 3 \
      --enc_in 56 \
      --dec_in 56 \
      --c_out 56 \
      --des 'Exp' \
      --d_model 256 \
      --d_ff 128 \
      --batch_size 16 \
      --dropout 0.1 \
      --learning_rate 0.001 \
      --train_epochs 20 \
      --patience 3 \
      --alpha 0.9 \
      --n_heads 8
done