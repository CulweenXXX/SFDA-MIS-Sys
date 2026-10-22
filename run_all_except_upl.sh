#!/bin/bash
# 串行执行除 UPL 外的所有 K-fold 脚本
set -e

cd "$(dirname "$0")"

for script in run_kfold_tent.sh run_kfold_mt.sh run_kfold_procontrast.sh run_kfold_iplc.sh; do
    echo "========== ${script} =========="
    bash "${script}"
done
