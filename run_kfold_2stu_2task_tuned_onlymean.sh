#!/bin/bash
# 用法: bash run_kfold_2stu_2task_tuned_onlymean.sh
# fold 0/1/2/3 分别在 GPU 0/1/2/3 上并行跑，都结束后 fold 4 在 GPU 0 上跑

cd "$(dirname "$0")"
export PYTHONPATH=.

TARGET=D
NOTE=MT98_双学生_2task_tuned_onlyMean_noSkip
EPOCH=20
INTERVAL_VALID=5
FOLD_JSON=./kfold_splits/mms/kfold_${TARGET}.json
MODEL_FILE=logs_train/mms/A/source_model_mms.tar

LOG_DIR=logs_target/mms/${TARGET}/${NOTE}
mkdir -p "${LOG_DIR}"

# run_fold <fold编号> <GPU编号>
run_fold() {
    fold=$1
    gpu=$2
    echo "fold ${fold} -> GPU ${gpu}"
    python method/mms/meanpl.py \
        --target "${TARGET}" \
        --gpu "${gpu}" \
        --model-file "${MODEL_FILE}" \
        --fold-json "${FOLD_JSON}" \
        --fold-id "${fold}" \
        --note "${NOTE}" \
        --epoch "${EPOCH}" \
        --interval-valid "${INTERVAL_VALID}" \
        --pl-source mean \
        --skip-empty-slices False
}


run_fold 0 0 &
run_fold 1 1 &
run_fold 2 2 &
run_fold 3 3 &
wait

run_fold 4 0 &
wait
