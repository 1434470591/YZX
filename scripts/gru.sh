#!/bin/bash
# This script runs the Python script in the scripts directory
# and then returns to the root directory.
cd ..
model_name=GRU

start_time=$(date +%s)

python -u main.py\
    --task_name Forecasting \
    --is_training 1 \
    --model $model_name \
    --data QuaDriGa \
    --features M \
    --output_size 96 \
    --input_size 96 \
    --hidden_size 192 \
    --num_layers 4 \
    --train_epochs 500 \
    --learning_rate 1e-4 \
    --des 'Exp' \
    --itr 1 \
    --gpu 3 \

end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

printf "The script has finished running. Cost time: %02dh %02dm %02ds\n" $hours $minutes $seconds