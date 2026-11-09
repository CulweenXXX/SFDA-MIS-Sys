#!/bin/bash
# 执行 B、C 域 MMS 数据集的 K-fold 测试
# 方法: CCRC / IPLC / MT / MT98_2stu_2task / Tent
#
# 用法: bash run_all_except_upl.sh
# 可用环境变量覆盖:
#   GPUS="0 1 2 3"   使用的 GPU 编号（按顺序轮转分配给各 fold）
#   FOLDS="0 1 2 3 4" 要跑的 fold 编号
#   PYTHON=python3    解释器

cd "$(dirname "$0")"
export PYTHONPATH=.

PYTHON=${PYTHON:-python}
GPUS=(${GPUS:-0 1 2 3})
FOLDS=(${FOLDS:-0 1 2 3 4})
TARGETS=(B C D)
MODEL_FILE=logs_train/mms/A/source_model_mms.tar

# run_folds <脚本名> <note> [额外参数...]
# 把 FOLDS 依次分配到 GPUS 上并行执行；一轮 GPU 占满后 wait，再跑下一轮
run_folds() {
    local script=$1
    local note=$2
    shift 2

    local gpu_idx=0
    local running=0
    for fold in "${FOLDS[@]}"; do
        local gpu=${GPUS[$gpu_idx]}
        echo "  [${TARGET}] fold ${fold} -> GPU ${gpu} (${script})"
        "$PYTHON" "method/mms/${script}" \
            --target "${TARGET}" \
            --gpu "${gpu}" \
            --model-file "${MODEL_FILE}" \
            --fold-json "${FOLD_JSON}" \
            --fold-id "${fold}" \
            --note "${note}" \
            "$@" &
        running=$((running + 1))
        gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
        if (( gpu_idx == 0 )); then
            wait
            running=0
        fi
    done
    if (( running > 0 )); then
        wait
    fi
}

for TARGET in "${TARGETS[@]}"; do
    FOLD_JSON=./kfold_splits/mms/kfold_${TARGET}.json
    echo "=================================================="
    echo "  Target domain: ${TARGET}  (${FOLD_JSON})"
    echo "=================================================="

    # CCRC
    echo "========== CCRC (${TARGET}) =========="
    run_folds ccrc.py "CCRC_noSkip" \
        --epoch 20 --interval-valid 5 \
        --lr 5e-4 --model-ema-rate 0.98 \
        --skip-empty-slices False

    # IPLC
    # echo "========== IPLC (${TARGET}) =========="
    # run_folds iplc.py "IPLC_noSkip" \
    #     --epoch 20 --interval-valid 5 \
    #     --lr 1e-6 \
    #     --skip-empty-slices False

    # MT
    echo "========== MT (${TARGET}) =========="
    run_folds mt.py "MT_noSkip" \
        --epoch 20 --interval-valid 5 \
        --lr 5e-4 --model-ema-rate 0.98 --pseudo-label-threshold 0.75 \
        --skip-empty-slices False

    # MT98_2stu_2task (meanpl.py)
    echo "========== MT98_2stu_2task (${TARGET}) =========="
    run_folds meanpl.py "MT98_双学生_2task_tuned_onlyMean_noSkip" \
        --epoch 20 --interval-valid 5 \
        --pl-source mean \
        --skip-empty-slices False

    # Tent
    echo "========== Tent (${TARGET}) =========="
    run_folds tent.py "Tent_noSkip" \
        --optimizer SGD --lr 1e-3 --momentum 0.9 --steps 1 \
        --epoch 1 --interval-valid 1 \
        --skip-empty-slices False
done

echo "All done: targets=${TARGETS[*]}, methods=CCRC,IPLC,MT,MT98_2stu_2task,Tent"
