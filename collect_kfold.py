# collect_kfold.py
import re, os
import numpy as np

TARGET = 'D'
NOTE = 'MT98_双学生_CE无mask_MSE一致hard'
K = 5
BASE = f'./logs_target/mms/{TARGET}'

def extract_last_student_test_dice(result_file):
    """从 result.txt 里抓最后一个 Student Test 的 avg dice"""
    with open(result_file) as f:
        content = f.read()
    # 匹配 'Student Test N dice: ... avg:XX.XX'
    matches = re.findall(
        r'Teacher Test \d+ dice:.*?avg:([\d.]+)', content)
    return float(matches[-1]) if matches else None

dices_per_fold = []
logs = ""

for fold in range(K):
    result_file = os.path.join(BASE, f'{NOTE}_fold{fold}', 'result.txt')
    if not os.path.exists(result_file):
        print(f'[missing] {result_file}')
        continue
    d = extract_last_student_test_dice(result_file)
    if d is not None:
        dices_per_fold.append(d)
        logs += f'Fold {fold}: {d:.2f} \n'
        # print(f'Fold {fold}: {d:.2f}')

dices = np.array(dices_per_fold)
logs += (
    f"\n=== {K}-Fold CV ({TARGET}) ===\n"
    f"Mean: {dices.mean():.2f}\n"
    f"Std : {dices.std():.2f}\n"
    f"Min/Max: {dices.min():.2f} / {dices.max():.2f}\n"
)

print(logs)
with open(os.path.join(BASE,NOTE+'_5折交叉验证'),'w') as f:
    f.write(logs)