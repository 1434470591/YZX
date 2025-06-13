#!/bin/bash
# This script runs the Python script in the scripts directory
# and then returns to the root directory.
cd ..
model_name=LLMs

start_time=$(date +%s)

for llm_type in 'gpt2'
# for llm_type in 'llama' 'gpt2-large' 'gpt2-medium' 'gpt2'
do
# for lradj in  'constant' 'cosine' 'type3
for lradj in  'cosine'
do
python -u main.py\
    --task_name Forecasting \
    --is_training 0 \
    --model $model_name \
    --data QuaDriGa \
    --SNR_ID 100\
    --features M \
    --llm_type $llm_type \
    --llm_layers 6 \
    --llm_d_model 768 \
    --llm_d_ff 768 \
    --des 'Exp' \
    --itr 1 \
    --gpu 0 \
    --train_epochs 500 \
    --batch_size 1024 \
    --lradj $lradj \
    --learning_rate 5e-4 \
    --speculative 1 \


done
done

end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

printf "The script has finished running. Cost time: %02dh %02dm %02ds\n" $hours $minutes $seconds
