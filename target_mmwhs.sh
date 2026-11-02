#!/bin/bash
# 用法: bash target_mmwhs.sh
# 当前 mmwhs 目录下的 3 个方法: MT / Tent / CCRC
# 每次两个方向并行: mr->ct (GPU 2), ct->mr (GPU 3)

cd "$(dirname "$0")"
export PYTHONPATH=.

DATA_DIR=/data0/grchen/Data/mmwhs
CT_WEIGHT=/data0/grchen/SFDA_MIS_TestSys/logs_train/mmwhs/ct/source_model_mmwhs_ct.pth.tar
MR_WEIGHT=/data0/grchen/SFDA_MIS_TestSys/logs_train/mmwhs/mr/source_model_mmwhs_mr.pth.tar

run_pair() {
    # run_pair <method.py> <note前缀> <额外参数...>
    local script=$1
    local prefix=$2
    shift 2

    python "method/mmwhs/${script}" --data-dir "${DATA_DIR}" \
        --source mr --target ct --gpu 2 --model-file "${MR_WEIGHT}" \
        --note "${prefix}_mr2ct" --lr 5e-6 "$@" &
    python "method/mmwhs/${script}" --data-dir "${DATA_DIR}" \
        --source ct --target mr --gpu 3 --model-file "${CT_WEIGHT}" \
        --note "${prefix}_ct2mr" --lr 5e-6 "$@" &
    wait
}

# MT
run_pair mt.py MT --model-ema-rate 0.98

# Tent
run_pair tent.py Tent --epoch 5 --interval-valid 1

# CCRC
run_pair ccrc.py CCRC --model-ema-rate 0.98
