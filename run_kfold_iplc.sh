#!/bin/bash
# 用法: bash run_kfold_iplc.sh
# fold 0/1/2/3 分别在 GPU 0/1/2/3 上并行跑，都结束后 fold 4 在 GPU 0 上跑

cd "$(dirname "$0")"
export PYTHONPATH=.

TARGET=D
FOLD_JSON=./kfold_splits/mms/kfold_${TARGET}.json
NOTE=IPLC

# run_fold <fold编号> <GPU编号>
run_fold() {
    fold=$1
    gpu=$2
    echo "fold ${fold} -> GPU ${gpu}"
    python method/iplc.py \
        --target "${TARGET}" \
        --gpu "${gpu}" \
        --model-file logs_train/mms/A/source_model_mms.tar \
        --fold-json "${FOLD_JSON}" \
        --fold-id "${fold}" \
        --note "${NOTE}" \
        --epoch 20 \
        --interval-valid 5 \
        --lr 1e-6
}

run_fold 0 0 &
run_fold 1 1 &
run_fold 2 2 &
run_fold 3 3 &
run_fold 4 0 &
wait


