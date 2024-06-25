#!/bin/bash

for i in {0..4}
do
  python ./leave_one_out.py --i "$i" --strat YOTO
done