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


BODY = (
    r' - LV: ([\d.]+)±[\d.]+'
    r' MYO: ([\d.]+)±[\d.]+'
    r' RV: ([\d.]+)±[\d.]+'
    r'.*?Mean \(LV,MYO,RV\): ([\d.]+)'
)


def extract_metrics(content, prefix):
    """按 prefix 抓最后一个匹配的 LV/MYO/RV/Mean 四个 avg Dice"""
    matches = re.findall(prefix + BODY, content)
    if matches:
        return tuple(float(v) for v in matches[-1])
    return None


def extract_last_test_dice(content):
    """抓最后一个 Test（MT 优先 Student Test）的 Dice"""
    for prefix in (r'CL Best Test \d+ Dice', r'Student Test \d+ Dice', r'Test \d+ Dice'):
        values = extract_metrics(content, prefix)
        if values is not None:
            return values
    return None


def extract_source_only_dice(content):
    """抓 Source Only 的 Dice（没有 target 自适应时的源域性能）"""
    return extract_metrics(content, r'Source Only \d+ Dice')


def fmt_row(tag, values):
    lv, myo, rv, mean_dice = values
    return f'{tag}: LV {lv:.2f}  MYO {myo:.2f}  RV {rv:.2f}  Mean {mean_dice:.2f}\n'


def aggregate(values_list):
    arr = np.array(values_list)
    return (
        f"LV : {arr[:, 0].mean():.2f}±{arr[:, 0].std():.2f}\n"
        f"MYO: {arr[:, 1].mean():.2f}±{arr[:, 1].std():.2f}\n"
        f"RV : {arr[:, 2].mean():.2f}±{arr[:, 2].std():.2f}\n"
        f"Avg: {arr[:, 3].mean():.2f}±{arr[:, 3].std():.2f}\n"
    )


per_fold = []
source_per_fold = []
for fold in range(K):
    result_file = os.path.join(BASE, f'fold_{fold}', 'result.txt')
    if not os.path.exists(result_file):
        print(f'[missing] {result_file}')
        continue
    with open(result_file) as f:
        content = f.read()
    values = extract_last_test_dice(content)
    if values is None:
        print(f'[no Test dice] {result_file}')
    else:
        per_fold.append((fold, values))
    source_values = extract_source_only_dice(content)
    if source_values is None:
        print(f'[no Source Only dice] {result_file}')
    else:
        source_per_fold.append((fold, source_values))

if not per_fold and not source_per_fold:
    print('no data collected')
    sys.exit(1)

logs = ''
if source_per_fold:
    logs += '--- Source Only per fold ---\n'
    for fold, values in source_per_fold:
        logs += f'Fold {fold} ' + fmt_row('SourceOnly', values)
    logs += (
        f"\n=== {K}-Fold CV Source Only ({TARGET}) ===\n"
        + aggregate([v for _, v in source_per_fold])
    )

if per_fold:
    logs += '\n--- Adaptation per fold ---\n'
    for fold, values in per_fold:
        logs += f'Fold {fold} ' + fmt_row(NOTE, values)
    logs += (
        f"\n=== {K}-Fold CV ({TARGET}, {NOTE}) ===\n"
        + aggregate([v for _, v in per_fold])
    )
print(logs)

out_file = os.path.join(BASE, f'{K}折交叉验证.txt')
with open(out_file, 'w') as f:
    f.write(logs)
print(f'saved: {out_file}')
