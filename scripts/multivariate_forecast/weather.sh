#export CUDA_VISIBLE_DEVICES=1

model_name=DSIN_PMA


for pred_len in 96 192 336 720
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_96_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len $pred_len \
        --e_layers_time 1 \
        --e_layers_variate 3 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 256 \
        --dropout 0.1 \
        --train_epochs 20 \
        --patience 3 \
        --n_heads 8 \
        --batch_size 8 \
        --alpha 0 \
        --learning_rate 0.001
done

