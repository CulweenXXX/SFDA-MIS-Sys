#!/bin/bash

# PYTHONPATH=. python method/mmwhs/tent.py --source mr --target ct --gpu 0 --note Tent_mr2ct_normal --normal True --lr 1e-5 &
# PYTHONPATH=. python method/mmwhs/tent.py --source ct --target mr --gpu 0 --note Tent_ct2mr_normal --normal True --lr 1e-5 &

# PYTHONPATH=. python method/mmwhs/tent.py --source mr --target ct --gpu 1 --note Tent_mr2ct --lr 1e-5 &
# PYTHONPATH=. python method/mmwhs/tent.py --source ct --target mr --gpu 1 --note Tent_mr2ct --lr 1e-5 &
# wait 

PYTHONPATH=. python method/mmwhs/mt.py --source mr --target ct --gpu 2 --note MT_mr2ct_normal_wT75 --normal  --lr 5e-6 --pseudo-label-threshold 0.75 &
PYTHONPATH=. python method/mmwhs/mt.py --source ct --target mr --gpu 3 --note MT_ct2mr_normal_wT75 --normal  --lr 5e-4 --pseudo-label-threshold 0.75 &
wait

PYTHONPATH=. python method/mmwhs/mt.py --source mr --target ct --gpu 2 --note MT_mr2ct_wT75 --lr 5e-6 --pseudo-label-threshold 0.75 &
PYTHONPATH=. python method/mmwhs/mt.py --source ct --target mr --gpu 3 --note MT_mr2ct_wT75 --lr 5e-4 --pseudo-label-threshold 0.75 &
wait 

PYTHONPATH=. python method/mmwhs/mt.py --source mr --target ct --gpu 2 --note MT_mr2ct_normal_wT90 --normal  --lr 5e-6 --pseudo-label-threshold 0.9 &
PYTHONPATH=. python method/mmwhs/mt.py --source ct --target mr --gpu 3 --note MT_ct2mr_normal_wT90 --normal  --lr 5e-4 --pseudo-label-threshold 0.9 &
wait

PYTHONPATH=. python method/mmwhs/mt.py --source mr --target ct --gpu 2 --note MT_mr2ct_wT90 --lr 5e-6 --pseudo-label-threshold 0.9 &
PYTHONPATH=. python method/mmwhs/mt.py --source ct --target mr --gpu 3 --note MT_mr2ct_wT90 --lr 5e-4 --pseudo-label-threshold 0.9 &
wait 