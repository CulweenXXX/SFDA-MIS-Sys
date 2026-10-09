# 数据集介绍

本目录说明 **SFDA_MIS_TestSys**（Source-Free Domain Adaptation for Medical Image Segmentation Test System）使用的三个医学图像分割数据集。所有原始数据统一存放在 `/data0/grchen/Data/` 下，分别对应 2D 眼底、心脏 MRI 与脑肿瘤 MRI 三类典型任务。

| 数据集 | 模态 | 任务 | 域 / 中心 | 类别 | 本地路径 |
| --- | --- | --- | --- | --- | --- |
| Fundus | 2D 彩色眼底照片 | 视盘 / 视杯分割 | 4（+1）个来源域 | disc / cup | `/data0/grchen/Data/Fundus` |
| M&MS | 短轴心脏 cine-MRI | 心脏结构分割 | A / B / C / D 四个扫描仪中心 | LV / MYO / RV | `/data0/grchen/Data/M&MS` |
| BraTS 2024 Glioma | 多参数脑 MRI（4 模态） | 治疗后胶质瘤亚区分割 | 7 家机构（单库多中心） | NETC / SNFH / ET / RC | `/data0/grchen/Data/BraTs2024_Glioma` |

> 说明：M&MS 与 BraTS 在源域无监督（SFDA）实验中的常用处理为“单个源域 + 多个无标注目标域”，下文分别给出本地数据的域划分、格式、标注编码与推荐用法。

---

## 1. Fundus —— 眼底视盘/视杯分割

### 1.1 任务背景

青光眼是全球第二大致盲眼病，临床上常通过眼底照片中**视盘（Optic Disc, OD）**与**视杯（Optic Cup, OC）**的形态（如杯盘比 CDR）进行筛查。本数据集用于视杯/视盘语义分割，是域泛化/源域无监督域适应（SFDA）文献中的常用基准（即 DPL 等工作中使用的 `disc_cup_split` 划分）。

### 1.2 域组成与划分

基准共登记 5 个来源域（本地已下载 Domain1–Domain4，Domain5 未下载）：

| 域 | 来源数据集 | 说明 | 训练 / 测试 |
| --- | --- | --- | --- |
| Domain1 | Drishti-GS | 印度 Aravind 眼科医院采集，单一印度人群 | 50 / 51（共 101） |
| Domain2 | RIM-ONE r3 | 西班牙，3 位专家标注的参考标准 | 99 / 60（共 159） |
| Domain3 | REFUGE（training） | MICCAI 2018 青光眼挑战赛训练集 | 320 / 80（共 400） |
| Domain4 | REFUGE（validation） | REFUGE 验证集，作为独立域使用 | 320 / 80（共 400） |
| Domain5 | IDRiD（ISBI 2018） | 印度糖尿病视网膜病变数据集 | 81（仅登记，本地未下载） |

### 1.3 本地目录结构

```text
Fundus/                             # 约 5.1 GB
├── readme.md                       # 原始来源登记（域来源与图像数）
├── Domain1/                        # Drishti-GS
│   ├── train/
│   │   ├── image/                  # 50 张，.png，RGB，原始分辨率（约 2045×1752）
│   │   ├── mask/                   # 50 张，.png，8-bit 灰度，取值 {0,128,255}
│   │   └── ROIs/
│   │       ├── image/              # 50 张，.png，以视盘为中心裁剪的 800×800
│   │       └── mask/               # 50 张，.png，800×800
│   └── test/                       # 同上结构，51 张
├── Domain2/                        # RIM-ONE r3：image 为 .jpg，mask 为 .png
├── Domain3/                        # REFUGE training：image 为 .jpg，mask 为 .bmp
├── Domain4/                        # REFUGE validation：image 为 .jpg，mask 为 .bmp
├── Domain5/                        # （未下载）
└── REFUGE/
    └── train/ROIs/                 # REFUGE 训练集全量 400 张的 800×800 ROI 版本（未划分）
```

### 1.4 标注编码与预处理

- **标注编码**（`mask` 与 `image` 同名）：`0` = 视杯 cup（内部区域）、`128` = 视盘环 rim（即 disc \ cup）、`255` = 背景。
  - 视杯：`mask == 0`
  - 视盘：`mask == 0` 或 `mask == 128`（两者并集）
  - 该解析逻辑见 `SFDA-DPL-main/dataloaders/custom_transforms.py`（`Normalize_tf` → `to_multilabel`）。
- **ROIs**：以视盘为中心的 800×800 裁剪版本，多数 SFDA 方法在 ROI 上训练；`ROIs/image` 与 `ROIs/mask` 文件名一一对应。
- **归一化**：训练代码通常执行 `img / 127.5 - 1`，将图像映射到 `[-1, 1]`。
- **注意**：各域图像尺寸不统一（Domain1 ≈ 2045×1752、Domain2 2144×1424、Domain3 2124×2056、Domain4 1634×1634）；目录中的 `.DS_Store`、`._*` 为 macOS 冗余文件，读取时需过滤。

### 1.5 SFDA 使用建议

- **源域**：取某一 Domain 的 `train`（带标注）训练模型。
- **目标域**：其余 Domain 的 `train`（只用图像，标签不可见）做适配；各自的 `test`（带标注）用于评估。
- **常用指标**：视杯 Dice 与视盘 Dice；`metrics/metrics.py` 提供 2D 的 Dice / ASSD / HD95 实现。

### 1.6 参考

- Drishti-GS 官网：<http://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php>
- REFUGE 挑战赛官网：<https://refuge.grand-challenge.org/>
- IDRiD 官网：<https://idrid.grand-challenge.org/>
- 基准划分来源：Chen et al., *Source-Free Domain Adaptive Fundus Image Segmentation with Denoised Pseudo-Labeling*, MICCAI 2021（<https://arxiv.org/pdf/2109.09735.pdf>）

---

## 2. M&MS —— 多中心、多厂商、多疾病心脏 MRI 分割

### 2.1 任务背景

M&MS（Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation）挑战赛为 MICCAI 2020 赛事（成果发表于 IEEE TMI 2021），旨在评测模型在**不同临床中心、不同扫描仪厂商、不同疾病**之间的泛化能力。官方数据共 375 例短轴心脏 cine-MRI，来自 4 种扫描仪厂商、6 家医院、3 个国家（西班牙、加拿大、德国）。

分割目标为三个心脏结构：

| 标签值 | 结构 | 说明 |
| --- | --- | --- |
| 1 | LV | 左心室腔 |
| 2 | MYO | 左心室心肌 |
| 3 | RV | 右心室腔 |
| 0 | 背景 | — |

### 2.2 四个扫描仪中心

| 域 | 扫描仪厂商 | 本地角色 |
| --- | --- | --- |
| A | Siemens | 源域候选（train / valid / test 三个划分均有数据与标注） |
| B | Philips | 目标域候选（仅 train，无标注划分） |
| C | GE | 目标域候选（仅 train，无标注划分） |
| D | Canon | 目标域候选（仅 train，无标注划分） |

### 2.3 本地目录结构与统计

```text
M&MS/
├── train/
│   ├── img/{A,B,C,D}/              # *.nii.gz，256×256×N，float32，已归一化到 [-1,1]
│   ├── lab/{A,B,C,D}/              # <name>_gt.nii.gz，取值 {0,1,2,3}
│   └── fsm_mask/{B,C,D}/           # 目标域辅助浮点掩码（.npy），见 2.5
├── valid/
│   ├── img/A/                      # 仅 A 域（B/C/D 目录为空）
│   └── lab/A/                      # 带标注
└── test/
    ├── img/A/                      # 仅 A 域（B/C/D 目录为空）
    └── lab/A/                      # 带标注
```

本地统计（文件数 = 帧数；同一患者可有多帧，文件名后缀为帧号，如 `A0S9V9_0` / `A0S9V9_9`）：

| 划分 | A | B | C | D | 合计 |
| --- | --- | --- | --- | --- | --- |
| train | 135 帧 / 77 人 | 250 帧 / 125 人 | 150 帧 / 75 人 | 100 帧 / 50 人 | 635 帧 / 327 人 |
| valid | 19 帧 / 19 人 | — | — | — | 19 帧 |
| test | 38 帧 / 19 人 | — | — | — | 38 帧 |

> 注：官方 M&MS 共 375 例；本地为预处理后的重划分版本（A 域被拆分为 train/valid/test，B/C/D 仅保留 train），A 域 train 中有 19 位患者仅含 1 帧。

### 2.4 数据格式与标签

- **图像**：`train/img/<域>/<患者ID>_<帧号>.nii.gz`，三维体积 `256×256×N`（深度 N 约 6–16，常见为 10），float32，像素范围已归一化到 `[-1, 1]`（窗宽调整预处理）。
- **标签**：`train/lab/<域>/<患者ID>_<帧号>_gt.nii.gz`，与图像同尺寸，取值 `{0,1,2,3}` = {背景, LV, MYO, RV}。
- **valid / test 仅含 A 域**：适合作为“源域 = A，目标域 ∈ {B, C, D}”的 SFDA 设定；目标域 B/C/D 的 `train/lab` 提供了完整 GT，可用于适配结果的直接评估。

### 2.5 fsm_mask 辅助掩码

`train/fsm_mask/{B, C, D}/` 为与目标域 B/C/D 一一对应的附加掩码：

- 文件名：`<患者ID>_<帧号>.npy`（与 `train/img` 对应）；
- 内容：float32 数组，形状 `(10, 224, 224)`，为**浮点型辅助掩码**（非离散类别标签）；
- 用途：随预处理/适配实验生成，具体含义请结合生成脚本使用，避免当作分割 GT。

### 2.6 切片级版本 `processed_mms`（可选）

`/data0/grchen/Data/processed_mms/`（约 4.6 GB）由 `/data0/grchen/dataset_test/mms_deal.py` 从 `Data/M&MS` 转换而来，便于逐切片训练：

```text
processed_mms/
├── train/img/<域>/<患者ID>_<帧号>_<切片号>.npy   # (1, 256, 256)，像素范围 [-1,1]
├── train/lab/<域>/<患者ID>_<帧号>_<切片号>.npy   # (1, 256, 256)，取值 {0,1,2,3}
├── valid/...
└── test/...
```

- 训练集体积统一为 10 层：不足 10 层补零，超过 10 层随机截取连续 10 层；valid / test 保留原始深度（可能不足或超过 10 层）；
- 原脚本中**含前景的切片文件名带 `_empty` 后缀**（命名遗留，读取时需按自身数据集类逻辑区分）。

### 2.7 SFDA 使用建议

- **源域**：A 域的 `train`（可另用 `valid` 选模型）；**目标域**：B/C/D 的 `train` 图像。
- **评估**：在目标域 `train/lab` 上计算 Dice / HD（`metrics/metrics.py` 的 2D 指标可逐切片使用）。
- 目标域 B/C/D 没有独立 `test` 划分，如需严格留出测试集，需自行按患者重新划分（保持患者级划分，避免同患者多帧跨集泄漏）。

### 2.8 参考

- 挑战赛官网（Universitat de Barcelona）：<https://www.ub.edu/mnms/>（页面可能暂时不可访问）
- 挑战赛论文：Campello et al., *Multi-Centre, Multi-Vendor and Multi-Disease Cardiac Segmentation: The M&MS Challenge*, IEEE TMI 2021, DOI: 10.1109/TMI.2021.3090082（PubMed: 34138702）

---

## 3. BraTS 2024 Glioma —— 治疗后胶质瘤 MRI 分割

### 3.1 任务背景

BraTS 2024 是脑肿瘤分割挑战赛（MICCAI 2024）中**首次聚焦治疗后（post-treatment）弥漫性胶质瘤**的分割任务，由 7 家机构贡献约 2200 例回顾性数据。官方使用 4 个多参数 MRI（mpMRI）序列，分割 4 个肿瘤亚区：

| 标签值 | 亚区 | 说明 |
| --- | --- | --- |
| 1 | NETC | 非强化肿瘤核心（necrosis/cysts 等） |
| 2 | SNFH | 周围非强化 FLAIR 高信号（水肿、浸润、治疗后改变等） |
| 3 | ET | 强化组织（active tumor / nodular enhancement） |
| 4 | RC | 切除腔（新近或陈旧） |
| 0 | 背景 | — |

官方预处理流程：DICOM → NIfTI（dcm2niix）→ HD-BET 颅骨剥离 → 仿射配准到 MNI 空间，最终为 1 mm 各向同性的 `182×218×182` 体积；官方训练/验证/测试按 70%/10%/20% 划分，评价指标为 lesion-wise Dice 与 lesion-wise HD95。

本地路径：`/data0/grchen/Data/BraTs2024_Glioma`（注意目录名实际为下划线 `_`，非 `-`）。

### 3.2 本地数据与统计

```text
BraTs2024_Glioma/
├── BraTS-GLI-00005-100/
│   ├── BraTS-GLI-00005-100-t1n.nii     # T1（pre-contrast）
│   ├── BraTS-GLI-00005-100-t1c.nii     # T1-Gd（contrast-enhanced）
│   ├── BraTS-GLI-00005-100-t2w.nii     # T2-weighted
│   ├── BraTS-GLI-00005-100-t2f.nii     # T2-FLAIR
│   └── BraTS-GLI-00005-100-seg.nii     # 分割标签，取值 {0,1,2,3,4}
├── BraTS-GLI-00005-101/                # 同一患者不同随访时间点
└── ...                                 # 共 700 个 case 目录，约 150 GB
```

| 统计项 | 数值 |
| --- | --- |
| case 目录数 | 700 |
| 唯一患者数（`BraTS-GLI-XXXXX`） | 272 |
| 随访时间点（文件夹名后缀） | 100–109，其中 100 为基线 |
| 多时间点患者数 | 198（纵向数据） |
| 每个 case 文件数 | 5（4 模态 + seg） |
| 体积尺寸 | 182×218×182（1 mm） |

> 本地为官方数据的 700 例子集（官方完整训练集规模更大）。同一患者存在多个时间点，使用随机划分时务必以**患者为单位**划分，防止同一患者的不同时相同时出现在源域与目标域造成信息泄漏。

### 3.3 SFDA 使用建议

- 可构造的域划分示例：按时间点（如源域 = 基线 100，目标域 = 随访 101–109）、按机构或按患者划分；
- 评估：`metrics/metrics.py` 中 `dice_brats_onehot` / `assd_brats_onehot` / `hd95_brats_onehot` 会把 5 通道预测与标签做 one-hot 对齐，返回 **(NETC, SNFH, ET, RC)** 四个类的逐样本指标（空目标返回 NaN）。

### 3.4 参考

- 官方数据与比赛页面（Synapse）：<https://www.synapse.org/Synapse:syn53708249/wiki/627500>
- 挑战赛论文：Correia de Verdier et al., *The 2024 Brain Tumor Segmentation (BraTS) Challenge: Glioma Segmentation on Post-treatment MRI*, arXiv:2405.18368（<https://arxiv.org/abs/2405.18368>）
- 官方指标实现：<https://github.com/rachitsaluja/BraTS-2023-Metrics>

---

## 4. 评估指标

`metrics/metrics.py` 提供以下实现：

| 函数 | 说明 |
| --- | --- |
| `compute_dice_coefficient(mask_gt, mask_pred)` | 逐样本 Dice（`[B,H,W]`，空掩码返回 NaN） |
| `compute_assd_coefficient(..., spacing_mm=(1.0,1.0))` | 平均对称表面距离（ASSD） |
| `compute_hd95_coefficient(..., spacing_mm=(1.0,1.0))` | 95% Hausdorff 距离（HD95） |
| `dice_brats_onehot` / `assd_brats_onehot` / `hd95_brats_onehot` | BraTS 5 通道 one-hot 评估，返回 4 个肿瘤亚区指标 |

约定：Dice ↑ 越大越好；ASSD / HD95 ↓ 越小越好。Fundus 与 M&MS 使用二分类/多分类的 2D 指标（可逐切片计算），BraTS 使用 one-hot 包装的 4 类指标。

---

## 5. 相关脚本

| 路径 | 用途 |
| --- | --- |
| `/data0/grchen/dataset_test/mms_deal.py` | M&MS：nii 体积 → `processed_mms` 逐切片 `.npy` |
| `/data0/grchen/dataset_test/mms_info*.py`、`mms_processed_visualize.py` | M&MS 数据检查与可视化 |
| `/data0/grchen/dataset_test/fundus_info.py` | Fundus ROI 数据检查 |
| `/data0/grchen/QXF_MMT/dataloaders/mms_dataloader*.py` | M&MS 训练用 Dataset（winadj + 2.5D） |
| `/data0/grchen/QXF_MMT/dataloaders/fundus_dataloader.py` | Fundus 训练用 Dataset |
| `/data0/grchen/QXF_MMT/dataloaders/brats2024_dataloader*.py` | BraTS 2024 训练用 Dataset |
| `/data0/grchen/SFDA-DPL-main/dataloaders/custom_transforms.py` | Fundus 掩码解析（0/128/255 → cup/disc） |

---

## 6. 参考资料与检索入口

- 官网 / 官方页面：见各节“参考”小节（Drishti-GS、REFUGE、IDRiD、M&MS、BraTS 2024 Synapse）。
- 知乎、CSDN 上有大量对上述数据集的中文整理（多为官方介绍的转述），可按需检索：
  - 知乎：<https://www.zhihu.com/search?type=content&q=BraTS%202024%20%E8%83%B6%E8%B4%A8%E7%98%A4>、<https://www.zhihu.com/search?type=content&q=%E5%BF%83%E8%84%8F%20M%26MS%20%E6%95%B0%E6%8D%AE%E9%9B%86>、<https://www.zhihu.com/search?type=content&q=%E7%9C%BC%E5%BA%95%20%E8%A7%86%E7%9B%98%20%E8%A7%86%E6%9D%AF%20%E5%88%86%E5%89%B2>
  - CSDN：<https://so.csdn.net/so/search?q=BraTS2024>、<https://so.csdn.net/so/search?q=M%26MS%20%E5%BF%83%E8%84%8F%E5%88%86%E5%89%B2>、<https://so.csdn.net/so/search?q=%E7%9C%BC%E5%BA%95%E8%A7%86%E7%9B%98%E8%A7%86%E6%9D%AF%E5%88%86%E5%89%B2>

使用数据集时请引用对应来源论文（见各节参考），并遵守各官方数据集的许可与使用条款。
