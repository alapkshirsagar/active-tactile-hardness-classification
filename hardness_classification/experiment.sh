#!/bin/bash

strategies=("random" "entropy" "variance" "YOTO")

for strat in "${strategies[@]}"
do
  for i in {0..4}
  do
    python3 ./leave_one_out.py --i "$i" --strat "$strat" 
  done
done

# Wait for all background processes to finish
wait