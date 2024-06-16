export CUDA_VISIBLE_DEVICES=0

model_name=DSIN_PMA

for pred_len in 96 192 336 720
do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id "ETTm1_96_96_$pred_len" \
        --model "$model_name" \
        --data ETTm1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers_time 2 \
        --e_layers_variate 2 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 256 \
        --batch_size 32 \
        --dropout 0.1 \
        --learning_rate 0.001 \
        --train_epochs 15 \
        --patience 3 \
        --n_heads 8 \
        --alpha 0.4
done


