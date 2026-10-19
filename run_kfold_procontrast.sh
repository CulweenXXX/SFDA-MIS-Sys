#!/bin/bash
# run_kfold_procontrast.sh

export PYTHONPATH=.
TARGET=D
K=5
FOLD_JSON=./kfold_splits/mms/kfold_${TARGET}.json
NOTE=ProtoContrast
GPU=3

for fold in $(seq 0 $((K-1))); do
    echo "=============================="
    echo "Running fold $fold"
    echo "=============================="
    python method/procontrast.py \
        --target $TARGET \
        --gpu $GPU \
        --model-file logs_train/mms/A/source_model_mms.tar \
        --fold-json $FOLD_JSON \
        --fold-id $fold \
        --note $NOTE \
        --pfa-epoch 5 \
        --cl-epoch 5 \
        --interval-valid 5
done
