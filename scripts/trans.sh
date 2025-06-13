#!/bin/bash
# This script runs the Python script in the scripts directory
# and then returns to the root directory.
cd ..
model_name=Transformer

start_time=$(date +%s)

for e_layers in 6 8
do
for d_layers in 6 8
do
python -u main.py\
    --task_name Forecasting \
    --is_training 1 \
    --model $model_name \
    --data QuaDriGa \
    --features M \
    --stack 0 \
    --enc_in 96 \
    --dec_in 96 \
    --c_out 96 \
    --d_model 1024 \
    --d_ff 1024 \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --train_epochs 500 \
    --batch_size 1024 \
    --learning_rate 1e-4 \
    --des 'Exp' \
    --itr 1 \
    --gpu 2 \

done
done
end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

printf "The script has finished running. Cost time: %02dh %02dm %02ds\n" $hours $minutes $seconds
