@echo off
setlocal enabledelayedexpansion
set n=10
set strategies=random entropy variance
set all_models=OFNN DinoNN ConvLSTM
set models=ConvLSTM
set dataset=hardness

for /L %%r in (1,1,%n%) do (
    echo Run #%%r
    for %%s in (%strategies%) do (
      for %%m in (%models%) do (
        for /L %%i in (0,1,4) do (
          set "comp_index=%%i"
          call python ./leave_one_out.py --comp_index !comp_index! --strat %%s --prefix "Final_exp_Aug_14" --model %%m --dataset !dataset!
        )
      )
    )
)
