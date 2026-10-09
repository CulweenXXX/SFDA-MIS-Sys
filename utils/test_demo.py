import torch
import numpy as np
from .metrics import dice_onehot, assd_onehot

B, C, H, W = 8, 5, 128, 128
NUM_CLASS = 5
names = ['BG', 'A', 'B', 'C', 'D']

# logits: [B, C, H, W]，浮点随机
pred = torch.randn(B, C, H, W)

# target: [B, 1, H, W]，整数 label 0~4
target = torch.randint(0, NUM_CLASS, (B, 1, H, W))

result = dice_onehot(pred, target, NUM_CLASS, names)

for key, value in result.items():
    print(f'{key}: {np.nanmean(value):.4f}')