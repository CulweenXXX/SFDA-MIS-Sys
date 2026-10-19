#!/bin/bash
# run_kfold_mt.sh

export PYTHONPATH=.
TARGET=D
K=5
FOLD_JSON=./kfold_splits/mms/kfold_${TARGET}.json
NOTE=MT
GPU=1

for fold in $(seq 0 $((K-1))); do
    echo "=============================="
    echo "Running fold $fold"
    echo "=============================="
    python method/mt.py \
        --target $TARGET \
        --gpu $GPU \
        --model-file logs_train/mms/A/source_model_mms.tar \
        --fold-json $FOLD_JSON \
        --fold-id $fold \
        --note $NOTE \
        --lr 5e-4 \
        --model-ema-rate 0.98 \
        --pseudo-label-threshold 0.75 \
        --epoch 20 \
        --interval-valid 1
done
