#!/bin/bash
# Number of times to repeat the whole set of experiments
n=10

strategies=("random" "entropy" "variance" "YOTO")

for (( run=1; run<=n; run++ ))
do
  echo "Run #$run"
  for strat in "${strategies[@]}"
  do
    for i in {0..4}
    do
      /home/junyi/miniconda3/envs/ip_hardness/bin/python /home/junyi/ip-hardness/hardness_classification/leave_one_out.py --comp_index "$i" --strat "$strat" --prefix "Experiment_"
    done
  done
done

# Wait for all background processes to finish
wait
