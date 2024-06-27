#!/bin/bash

sbatch -t 24:00:00 -c 1 --mem-per-cpu=4G -p gpu --gres=gpu:1 -C 'rtx3080|rtx3090' ./experiment.sh
