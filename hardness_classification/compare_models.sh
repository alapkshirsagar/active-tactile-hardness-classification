#!/bin/bash
n=10 # Number of times to repeat the whole set of experiments
i=0 # comp class does not matter

strategies=("YOTO")
models = ("OFNN" "DinoNN" "ConvLSTM")


for (( run=1; run<=n; run++ ))
do
  echo "Run #$run"
  for strat in "${strategies[@]}"
  do
    for model in "${models[@]}"
    do
      /home/junyi/miniconda3/envs/ip_hardness/bin/python /home/junyi/ip-hardness/hardness_classification/leave_one_out.py --i "$i" --strat "$strat" --prefix "Comp_Model_Aug10" --model "$model" 
    done
  done
done

# Wait for all background processes to finish
wait
