export CUDA_VISIBLE_DEVICES=0


model_name=DSIN_PMA


for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/PEMS/ \
      --data_path PEMS08.npz \
      --model_id PEMS08_96_$pred_len \
      --model $model_name \
      --data PEMS \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers_time 1 \
      --e_layers_variate 4 \
      --enc_in 170 \
      --des 'Exp' \
      --d_model 128 \
      --learning_rate 0.001 \
      --batch_size 4 \
      --train_epochs 15 \
      --alpha 0.2 \
      --dropout 0.1
done


