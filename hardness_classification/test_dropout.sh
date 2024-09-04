#!/bin/bash
export LC_NUMERIC="en_US.UTF-8" 

strat="variance"
model="DinoNN"
dataset="yuan"

dropout_values=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)


for dropout in "${dropout_values[@]}"; do
    for i in $(seq 0 8); do
        /home/junyi/miniconda3/envs/ip_hardness/bin/python  ./leave_one_out.py --comp_index $i --strat "variance" --prefix "Ablation_dropout" --model $model --dataset $dataset --dropout $dropout 
    done
done
