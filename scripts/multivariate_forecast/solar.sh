#export CUDA_VISIBLE_DEVICES=1


model_name=DSIN_PMA

for pred_len in 96 192 336 720
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/solar/ \
        --data_path solar_AL.csv \
        --model_id solar_96_96_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len $pred_len \
        --enc_in 137 \
        --des 'Exp' \
        --e_layers_time 2 \
        --e_layers_variate 3 \
        --d_model 128 \
        --dropout 0.1 \
        --n_heads 4 \
        --batch_size 8 \
        --train_epochs 10 \
        --alpha 0 \
        --learning_rate 0.001
done


