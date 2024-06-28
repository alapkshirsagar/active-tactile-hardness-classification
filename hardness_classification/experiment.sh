#!/bin/bash
python3.10 --version

strategies=("random" "entropy" "variance" "YOTO")

for strat in "${strategies[@]}"
do
  for i in {0..4}
  do
    /bin/python3 /home/junyi/ip-hardness/hardness_classification/leave_one_out.py --i "$i" --strat "$strat" 
  done
done

# Wait for all background processes to finish
wait