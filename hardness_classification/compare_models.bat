@echo off
set n=50
set i=0   

setlocal enabledelayedexpansion

set strategies=YOTO
set all_models=OFNN DinoNN ConvLSTM
set models=DinoNN
for /L %%r in (1,1,%n%) do (
    echo Run #%%r
    for %%s in (%strategies%) do (
        for %%m in (%models%) do (
            call python ./leave_one_out.py --comp_index !i! --strat %%s --prefix "Comp_Model_Aug_12" --model %%m
        )
    )
)
