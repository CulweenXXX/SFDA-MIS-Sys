import argparse
import os
import re
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='D', help='target domain (B/C/D)')
parser.add_argument('--note', type=str, default='IPLC', help='方法名，对应 logs_target 下的方法目录')
parser.add_argument('-k', '--folds', type=int, default=5, help='折数')
parser.add_argument('--root', type=str, default='./logs_target/mms', help='日志根目录')
args = parser.parse_args()

TARGET = args.target
NOTE = args.note
K = args.folds
BASE = os.path.join(args.root, TARGET, NOTE)


def extract_last_test_dice(result_file):
    """抓最后一个 Test（MT 优先 Student Test）的 LV/MYO/RV/Mean 四个 avg Dice"""
    with open(result_file) as f:
        content = f.read()
    body = (
        r' - LV: ([\d.]+)±[\d.]+'
        r' MYO: ([\d.]+)±[\d.]+'
        r' RV: ([\d.]+)±[\d.]+'
        r'.*?Mean \(LV,MYO,RV\): ([\d.]+)'
    )
    for prefix in (r'Student Test \d+ Dice', r'Test \d+ Dice'):
        matches = re.findall(prefix + body, content)
        if matches:
            return tuple(float(v) for v in matches[-1])
    return None


per_fold = []
for fold in range(K):
    result_file = os.path.join(BASE, f'fold_{fold}', 'result.txt')
    if not os.path.exists(result_file):
        print(f'[missing] {result_file}')
        continue
    values = extract_last_test_dice(result_file)
    if values is None:
        print(f'[no Test dice] {result_file}')
        continue
    per_fold.append((fold, values))

if not per_fold:
    print('no data collected')
    sys.exit(1)

logs = ''
for fold, (lv, myo, rv, mean_dice) in per_fold:
    logs += f'Fold {fold}: LV {lv:.2f}  MYO {myo:.2f}  RV {rv:.2f}  Mean {mean_dice:.2f}\n'

arr = np.array([v for _, v in per_fold])
logs += (
    f"\n=== {K}-Fold CV ({TARGET}, {NOTE}) ===\n"
    f"LV : {arr[:, 0].mean():.2f}±{arr[:, 0].std():.2f}\n"
    f"MYO: {arr[:, 1].mean():.2f}±{arr[:, 1].std():.2f}\n"
    f"RV : {arr[:, 2].mean():.2f}±{arr[:, 2].std():.2f}\n"
    f"Avg: {arr[:, 3].mean():.2f}±{arr[:, 3].std():.2f}\n"
)
print(logs)

out_file = os.path.join(BASE, f'{K}折交叉验证.txt')
with open(out_file, 'w') as f:
    f.write(logs)
print(f'saved: {out_file}')
