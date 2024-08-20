@echo off
setlocal enabledelayedexpansion
set n=5

set strategies=random
set all_models=OFNN DinoNN ConvLSTM
set models=DinoNN
set dataset=yuan
for /L %%r in (1,1,%n%) do (
    echo Run #%%r
    for %%s in (%strategies%) do (
      for %%m in (%models%) do (
        for /l %%i in (0,1,8) do (
          call python ./leave_one_out.py --comp_index %%i --strat %%s --prefix "Final_exp_Aug_12" --model %%m --dataset !dataset!
        )
      )
    )   
)