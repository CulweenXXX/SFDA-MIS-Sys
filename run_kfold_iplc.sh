#!/bin/bash
# 用法: bash run_kfold_iplc.sh
# 仅使用 GPU 0/1，每次最多并行跑 2 个 fold

cd "$(dirname "$0")"
export PYTHONPATH=.

TARGET=D
FOLD_JSON=./kfold_splits/mms/kfold_${TARGET}.json
NOTE=IPLC_noSkip

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
        --lr 1e-6 \
        --skip-empty-slices False
}

run_fold 0 0 & run_fold 1 1 & wait
run_fold 2 0 & run_fold 3 1 & wait
run_fold 4 0


