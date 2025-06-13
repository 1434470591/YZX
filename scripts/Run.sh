#!/bin/bash

start_time=$(date +%s)

echo "开始运行 rnn.sh \ lstm.sh \ gru.sh \ cnn.sh"
# bash gpt2.sh 
# bash trans.sh 
bash rnn.sh &>rnn.txt
bash lstm.sh &>lstm.txt
bash gru.sh &>gru.txt
bash cnn.sh &>cnn.txt
bash trans.sh &>trans.txt
bash llm.sh &>llm.txt

echo "所有脚本执行完毕。"

end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))
printf "The script has finished running. Cost time: %02dh %02dm %02ds\n" $hours $minutes $seconds