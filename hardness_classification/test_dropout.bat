@echo off

setlocal enabledelayedexpansion
set LC_NUMERIC=en-US

set "strat=variance"
set "model=DinoNN"
set "dataset=yuan"

rem Define the dropout values
set dropout_values=0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5

for %%d in (%dropout_values%) do (
    for /l %%i in (0,1,8) do (
        call python leave_one_out.py --comp_index %%i --strat !strat! --prefix "Sep_4_Ablation_Dropout" --model !model! --dataset !dataset! --dropout %%d
    )
)