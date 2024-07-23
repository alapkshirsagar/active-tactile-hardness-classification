#!/bin/bash
python3.10 --version

# Number of times to repeat the whole set of experiments
n=10

strategies=("random" "entropy" "variance" "YOTO")

for (( run=1; run<=n; run++ ))
do
  echo "Run #$run"
  for strat in "${strategies[@]}"
  do
    for i in {0..cls}
    do
      /bin/python3 /home/junyi/ip-hardness/hardness_classification/leave_one_out.py --i "$i" --strat "$strat" 
    done
  done
done

# Wait for all background processes to finish
wait
