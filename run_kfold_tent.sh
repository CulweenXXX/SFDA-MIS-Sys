#!/bin/bash
# run_kfold_tent.sh

export PYTHONPATH=.
TARGET=D
K=5
FOLD_JSON=./kfold_splits/mms/kfold_${TARGET}.json
NOTE=Tent
GPU=1

for fold in $(seq 0 $((K-1))); do
    echo "=============================="
    echo "Running fold $fold"
    echo "=============================="
    python method/tent.py \
        --target $TARGET \
        --gpu $GPU \
        --model-file logs_train/mms/A/source_model_mms.tar \
        --fold-json $FOLD_JSON \
        --fold-id $fold \
        --note $NOTE \
        --optimizer SGD \
        --lr 1e-3 \
        --momentum 0.9 \
        --steps 1 \
        --epoch 20 \
        --interval-valid 1
done
