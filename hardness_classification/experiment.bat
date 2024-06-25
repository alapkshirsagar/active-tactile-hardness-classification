for /L %%i in (0,1,4) do (
  call python ./leave_one_out.py --i %%i --strat random
)

for /L %%i in (0,1,4) do (
  call python ./leave_one_out.py --i %%i --strat entropy
)

for /L %%i in (0,1,4) do (
  call python ./leave_one_out.py --i %%i --strat variance
)

for /L %%i in (0,1,4) do (
  call python ./leave_one_out.py --i %%i --strat YOTO
)