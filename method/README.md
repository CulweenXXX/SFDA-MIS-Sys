# 方法复现说明（method/）

本目录在**统一的实验协议**下复现/适配了一批医学图像分割的**源域无监督域适应（Source-Free Domain Adaptation, SFDA）**方法，用于公平对比。每个方法一个可独立运行的 `method/*.py` 脚本，共享同一套数据加载、网络结构、K 折划分与评测口径。

## 已有方法一览

| 方法 | 脚本 | 论文来源 | 官方代码 | 状态 |
| --- | --- | --- | --- | --- |
| IPLC | `method/iplc.py` | MICCAI 2024（期刊扩展 IPLC+，JBHI 2025） | https://github.com/HiLab-git/IPLC | 已跑通 5 折 |
| UPL | `method/upl.py` | IEEE TMI 2023, arXiv:2309.10244 | https://github.com/HiLab-git/UPL-SFDA | 已实现，未跑 |
| Tent | `method/tent.py` | ICLR 2021, arXiv:2006.10726 | https://github.com/DequanWang/tent | 已跑通 5 折 |
| ProtoContrast | `method/procontrast.py` | MICCAI 2023, arXiv:2307.09769 | https://github.com/CSCYQJ/MICCAI23-ProtoContra-SFDA | 已跑通 5 折 |
| MT | `method/mt.py` | MICCAI 2023（CBMT），arXiv:2307.09973 | https://github.com/lloongx/SFDA-CBMT | 已跑通 5 折 |
| DPL | `method/dpl.py` | MICCAI 2021, arXiv:2109.09735 | https://github.com/cchen-cc/SFDA-DPL | 待实现（空文件占位） |
| PTBN | `method/ptbn.py` | arXiv:2006.10963（ICML 2020 UDL Workshop） | 无官方代码 | 待实现（空文件占位） |

---

## 0. 统一实验协议（所有方法共用）

**基础 pipeline**：源模型（域 A）→ 载入目标域待适应模型 → 用目标域 train 患者**无标注**图像做适应 → 每若干 epoch 在 val/test 上评测 → 汇总 5 折均值±std。

| 项目 | 设置 |
| --- | --- |
| 数据集 | M&MS 多中心心脏 MRI（域 A/B/C/D，4 类：BG / LV / MYO / RV），路径 `/data0/grchen/Data/M&MS`（详见 `dataloaders/README.md`） |
| 源域 | A；源模型 `logs_train/mms/A/source_model_mms.tar`（DeepLab checkpoint，含 `model_state_dict`），adaptation 阶段不再访问源数据/标签 |
| 目标域 | 默认 `--target D`，可切 B/C/D |
| 网络 | DeepLabv3+ / MobileNetV2，`output_stride=16`，SyncBN（`networks/deeplabv3.py`），`NUM_CLASSES=4` |
| 目标域划分 | 患者级 5 折交叉验证：`kfold_splits/mms/kfold_{B,C,D}.json`，每折 train:val:test ≈ 7:1:2，患者互不重叠（脚本内含重叠断言） |
| 训练数据 | 仅目标域 train 划分的**无标注**图像用于适应；val/test 的 GT 只用于评测 |
| 评测 | `utils/metrics.py`：切片拼回 volume，丢弃 GT 全空切片，按 volume 计算 Dice / ASSD / HD95 后取均值±std（口径与 CCRC 参考实现对齐） |
| 输出 | `logs_target/mms/{target}/{note}/fold_{i}/result.txt`（命令行、Source Only 基线、各阶段指标）与 `loss.txt` |

运行方式（脚本顶部可改 `TARGET` / `GPU` / `NOTE`）：

```bash
bash run_kfold_tent.sh          # Tent
bash run_kfold_iplc.sh          # IPLC（需要 SAM-Med2D，见 1.1）
bash run_kfold_procontrast.sh   # ProtoContrast
bash run_kfold_mt.sh            # MT
```

单折手动运行示例：

```bash
export PYTHONPATH=.
python method/tent.py --target D --gpu 0 \
    --model-file logs_train/mms/A/source_model_mms.tar \
    --fold-json ./kfold_splits/mms/kfold_D.json --fold-id 0 \
    --note Tent --epoch 20 --interval-valid 1
```

K 折汇总（读取各 fold 最后一个 Test 指标，写入 `logs_target/.../5折交叉验证.txt`）：

```bash
python collect_kfold.py --target D --note Tent
```

环境：Python 3.10 + PyTorch（本机 conda 环境 `Torch2.0` / `Torch2.2`），依赖 SimpleITK、surface_distance、opencv、scipy、numpy、pandas、matplotlib、yaml、einops；IPLC 额外需要 SAM-Med2D。

---

## 1. 已复现方法

### 1.1 IPLC — `method/iplc.py`

- **论文**：IPLC: Iterative Pseudo Label Correction Guided by SAM for Source-Free Domain Adaptation in Medical Image Segmentation，MICCAI 2024（扩展版 IPLC+，IEEE JBHI 2025）
- **官方仓库**：https://github.com/HiLab-git/IPLC
- **依赖资源**：SAM-Med2D 定制版 segment_anything（已放在 `external/segment_anything`，`predictor_sammed`）、SAM-Med2D 权重（默认参数指向 `/data0/grchen/IPLC/pretrain_model/sam-med2d_b.pth`，本地另有 `pre_trained_LLM/sam-med2d_b.pth`）
- **本地 pipeline**（每 epoch、每 batch）：
  1. DeepLab 前向得 softmax 概率 → 对各类取最大连通域 → 在前景内随机采样 3 个点作为 prompt（`--sample-times 10` 次重复采样，背景类不采样）；
  2. SAM-Med2D 以 prompt + 64×64 上采样概率图作为 `mask_input` 预测，聚合多次采样结果得到类别概率图；
  3. 由概率图熵生成权重图 `(2 − H) / 2`，argmax + 最大连通域得到修正伪标签；
  4. 损失 = 熵加权 Dice 损失（伪标签）+ `--curve-loss-weight 0.01` × 平均负曲率损失（mean negative curvature，促进边界平滑）；
  5. Adam 分组优化：encoder + ASPP 与 decoder（复现配置 `--lr 1e-5`，官方 1e-4；每 4 epoch ×0.9）；下一 epoch 用更新后的模型重新生成伪标签（对应 ICL 迭代修正的简化实现；复现配置 `--epoch 5`，官方 20）。
- **复现配置**：`run_kfold_iplc.sh` 使用 `--lr 1e-5 --epoch 5 --interval-valid 1`。官方 1e-4/20 epoch 在 DeepLabv3+/3D volume 设置下从第 1 个 epoch 后持续下降，D 域 5 折 Avg 81.21，低于 Source Only 83.26；降到 1e-5、5 epoch 后 D 域 5 折 Avg 85.08±1.42（Source Only 83.26±1.81），5 折均优于源模型。官方超参日志保留在 `logs_target/mms/D/IPLC_official_hparams/`。
- **说明**：官方实现为 UNet + 迭代内循环；本实现为统一 DeepLabv3+/M&MS 协议做了适配，熵加权与曲率监督与论文一致。

### 1.2 UPL — `method/upl.py`

- **论文**：UPL-SFDA: Uncertainty-aware Pseudo Label Guided Source-Free Domain Adaptation for Medical Image Segmentation，IEEE TMI 2023，arXiv:2309.10244
- **官方仓库**：https://github.com/HiLab-git/UPL-SFDA
- **本地 pipeline**：
  1. TDG（Target Domain Growing）：复制源模型 decoder 4 份（`UPLDeepLabV3Plus`）；
  2. 对同一输入做 0°/90°/180°/270° 旋转分别过 4 个 decoder，输出反旋转平均得集成预测；
  3. 平均预测阈值化（`pl_threshold=0.95`）+ 逐类最大连通域 → 伪标签；
  4. 损失 = 4 个预测头对伪标签的 Dice 损失均值 + 平均预测的熵最小化（隐式多头一致性约束）；
  5. Adam 分组优化 backbone + ASPP 与 4 个 decoder（默认 `--lr 1e-5`，每 4 epoch ×0.9，`--epoch 20`）。
- **状态**：代码已实现，但尚无 `run_kfold_upl.sh` 与 K 折日志，跑通流程与其他方法一致（把命令换成 `method/upl.py` 即可）。

### 1.3 Tent — `method/tent.py`

- **论文**：Tent: Fully Test-Time Adaptation by Entropy Minimization，ICLR 2021，arXiv:2006.10726
- **官方仓库**：https://github.com/DequanWang/tent（核心代码注释中标明 Reference）
- **本地 pipeline**：
  1. `configure_model`：模型转 train 模式、冻结全部参数，仅打开 norm 层（BN/SyncBN）的 affine weight/bias 梯度（`tent: adapt N / M params`）；
  2. 每 batch 前向 → softmax 熵损失（全像素平均）→ 反向 → 只更新 norm 仿射参数（`--steps 1`，默认 SGD lr 1e-3 / momentum 0.9，非 episodic）；
  3. 目标域 train 划分 shuffle 采样训练 20 个 epoch，每 epoch 评测。
- **本地差异**：官方原版在 eval 时用 batch 统计并禁用 running stats；此处因 SyncBN 实现与 ASPP 全局池化的限制，保留 running stats（且持续更新），见 `configure_model` 注释。

### 1.4 ProtoContrast — `method/procontrast.py`

- **论文**：Source-Free Domain Adaptation for Medical Image Segmentation via Prototype-Anchored Feature Alignment and Contrastive Learning，MICCAI 2023，arXiv:2307.09769
- **官方仓库**：https://github.com/CSCYQJ/MICCAI23-ProtoContra-SFDA
- **本地 pipeline**（单脚本内两阶段，`--pfa-epoch 5` + `--cl-epoch 5`）：
  1. 冻结分类器 `decoder.last_conv_2`，其权重 `view(4, -1)` 作为**源原型**；
  2. **PFA 阶段**：PCT（NeurIPS 2021）双向原型运输损失（`--nav-t 0.1`、`--proto-beta 0.001`、`--s-par 0.5`），目标特征与源原型双向对齐，Adam lr 1e-4；
  3. **CL 阶段**：以 PFA 结束的模型复制一份作为 frozen teacher（eval、不更新）；teacher 伪标签 + 熵分位掩码把像素分为低熵（anchor 正原型）与高熵（memory bank 负样本队列，`--queue-size 30000` / `--bg-queue-size 50000`），做原型对比损失（`--num-queries 64`、`--num-negatives 256`，可选 `--momentum-prototype`、`--use-source-prototypes`）；
  4. 可选熵过滤伪标签 CE（`--use-pseudo`，按 epoch 逐步放宽 `--drop-percent`）；Adam lr 1e-4。
- **本地映射**：对应官方仓库的 `train_target_adapt_PFA.yaml` / `train_target_adapt_CL.yaml` 两个 stage，网络与评测协议统一为 DeepLabv3+/M&MS。
- **本地差异（best-validation）**：按官方 PFA/CL 的 best-checkpoint 协议，每 epoch 用 val Dice 选择最优模型；PFA 最优模型用于初始化 CL 的 teacher/student；CL 连续 2 epoch 无提升则 early stop；最终报告 best checkpoint 的 test 指标。其他方法目前仍报告最后一个 epoch，跨方法对比时需注意这一差异。

### 1.5 MT — `method/mt.py`

- **参考论文**：Source-Free Domain Adaptive Fundus Image Segmentation with Class-Balanced Mean Teacher（CBMT），MICCAI 2023，arXiv:2307.09973
- **官方仓库**：https://github.com/lloongx/SFDA-CBMT
- **本地 pipeline**：
  1. teacher / student 均由源模型初始化（teacher 冻结梯度）；
  2. 同一 volume 的**弱增强**视图（仅 resize 等）输入 teacher，**强增强**视图（亮度扰动 + 高斯噪声，`strong_aug_woErase`）输入 student（`dataloaders/mms_dataloader.py` 的 `niiDataset_2transform_split` 返回双视图）；
  3. teacher 对弱视图预测取 argmax 作伪标签，监督 student 强视图（交叉熵）；
  4. 每个 batch 后按 EMA 更新 teacher：`θ_t = 0.98·θ_t + 0.02·θ_s`（`--model-ema-rate 0.98`）；
  5. Adam lr 5e-4，训练 20 epoch，val/test 同时输出 Teacher 与 Student 指标。
- **本地差异**：当前只启用基础伪标签损失（对应 CBMT 消融中的 P-L + EMA + Aug.）；`--pseudo-label-threshold 0.75` 与 `--mean-loss-calc-bound-ratio 0.2` 沿用了 CBMT 默认值，但阈值/掩码分支目前在代码中被注释，类别平衡 loss 校准尚未接入。

---

## 2. 待实现方法（空文件占位）

### 2.1 DPL — `method/dpl.py`

- **论文**：Source-Free Domain Adaptive Fundus Image Segmentation with Denoised Pseudo-Labeling，MICCAI 2021，arXiv:2109.09735
- **官方仓库**：https://github.com/cchen-cc/SFDA-DPL
- **计划 pipeline**：模型自身预测生成伪标签 → 像素级去噪（MC Dropout K 次前向的标准差不确定性过滤）+ 类别级去噪（特征到类原型的相对距离过滤）→ 仅用去噪后的可靠伪标签自训练。
- **备注**：眼底数据集的 `disc_cup_split` 划分与掩码解析已按 DPL 官方代码整理，见 `dataloaders/README.md`。

### 2.2 PTBN — `method/ptbn.py`

- **论文**：Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift（Nado et al.），arXiv:2006.10963，ICML 2020 UDL Workshop
- **计划 pipeline**：不更新任何参数，测试时用目标域 batch 重新统计 BN 层的 mean/var（即 eval 时启用 batch 统计），作为最简测试时适应 baseline（UPL-SFDA 论文中的对比方法之一）。

---

## 3. 目录说明

| 路径 | 用途 |
| --- | --- |
| `method/*.py` | 各方法的适应脚本（本页主角），参数风格统一 |
| `dataloaders/` | M&MS / Fundus 等数据集加载与增强（`niiDataset_2transform_split` 支持患者级 K 折与弱/强双视图），数据集细节见 `dataloaders/README.md` |
| `networks/` | DeepLabv3+（backbone：mobilenet/resnet/xception/drn）、SyncBN 等网络定义 |
| `utils/metrics.py` | volume 级 Dice / ASSD / HD95 评测与 `format_result` 日志格式化 |
| `kfold_splits/mms/` | 目标域 B/C/D 的患者级 5 折划分 JSON |
| `logs_train/` | 源模型权重（当前仅有 `mms/A/source_model_mms.tar`；`brats2024/`、`fundus/`、`mmwhs/` 为后续数据集占位） |
| `logs_target/` | 各方法适应日志与结果（`result.txt` / `loss.txt`），K 折汇总文件也写在这里 |
| `external/` | 外部依赖（SAM-Med2D 版 segment_anything、MedSAM2、SAM3），已被 `.gitignore` 忽略 |
| `pre_trained_LLM/` | 大模型预训练权重（SAM-Med2D `sam-med2d_b.pth`） |
| `collect_kfold.py` | 汇总指定方法 5 折的 Test Dice（表头优先取 `Student Test`） |
