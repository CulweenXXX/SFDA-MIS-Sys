import torch.utils.data as data
import os
import glob
import torch
import numpy as np
import albumentations as A

'''
contour_map = {
    "bg": 0,
    "lv_myo": 1,
    "la_blood": 2,
    "lv_blood": 3,
    "aa": 4,
}
'''

IMAGE_TAG = 'img'
LABEL_TAG = 'label'
MR_MIN, MR_MAX = -1.8, 4.4
CT_MIN, CT_MAX = -2.8, 3.2
TARGET_SIZE = 256

def build_transforms(is_train: bool, use_augment: bool = True):
    ops = []
    if is_train and use_augment:
        ops += [
            # 几何（合并原 Rotate + Affine）
            A.Affine(translate_percent=(-0.1, 0.1),
                     scale=(0.9, 1.1),
                     rotate=(-15, 15), p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            # 外观（[0,1] 语义下参数才是正确的）
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.03, 0.07), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            # 遮挡（0 在 [0,1] 下=背景，语义正确）
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 24),
                hole_width_range=(8, 24),
                fill=0, fill_mask=0, p=0.3,
            ),
        ]

    ops.append(A.Resize(TARGET_SIZE, TARGET_SIZE))
    return A.Compose(ops)

def build_adapt_transforms(is_train: bool):
    ops = []
    if is_train:
        ops += [
            # 几何（合并原 Rotate + Affine）
            # A.Affine(translate_percent=(-0.1, 0.1),
            #          scale=(0.9, 1.1),
            #          rotate=(-15, 15), p=0.5),
            # A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            # 外观（[0,1] 语义下参数才是正确的）
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.03, 0.07), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            # 遮挡（0 在 [0,1] 下=背景，语义正确）
            # A.CoarseDropout(
            #     num_holes_range=(1, 8),
            #     hole_height_range=(8, 24),
            #     hole_width_range=(8, 24),
            #     fill=0, fill_mask=0, p=0.3,
            # ),
        ]

    ops.append(A.Resize(TARGET_SIZE, TARGET_SIZE))
    return A.Compose(ops)

def collect_image_label_paths(rootdir, modality, phase):
    data_dir = os.path.join(rootdir, f'{modality}_{phase}')
    image_paths = sorted(glob.glob(os.path.join(data_dir, f'*{IMAGE_TAG}*.npy')))
    label_paths = sorted(glob.glob(os.path.join(data_dir, f'*{LABEL_TAG}*.npy')))
    return image_paths, label_paths


def to_unit_range(img: np.ndarray, modality: str) -> np.ndarray:
    """clip 到物理范围后映射到 [0,1]，保持非负，兼容 Gamma。"""
    lo, hi = (MR_MIN, MR_MAX) if modality == 'mr' else (CT_MIN, CT_MAX)
    img = np.clip(img, lo, hi).astype(np.float32)
    return (img - lo) / (hi - lo)

class MMWHSDataset(data.Dataset):
    def __init__(self, rootdir: str, modality: str, phase: str = 'train',
                 mode: str = 'source'):
        self.rootdir = rootdir
        self.modality = modality
        self.phase = phase
        self.mode = mode

        self.all_data_path, self.all_label_path = collect_image_label_paths(
            rootdir, modality, phase)

        # 只有 source 域的训练集做增强；验证/测试/target 只 Resize
        self.transform = build_transforms(
            is_train=(phase == 'train'),
            use_augment=(mode == 'source'),
        )

    def __getitem__(self, index):
        # ---- 读取 ----
        img = np.load(self.all_data_path[index])
        lab = np.load(self.all_label_path[index]).astype(np.int64)

        # ---- 统一成 [H,W]（单通道），避免 permute 维度坑 ----
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]

        # ---- 先到 [0,1]，再增强，最后映射到 [-1,1] ----
        img = to_unit_range(img, self.modality)             # [0,1]
        out = self.transform(image=img, mask=lab)
        img, lab = out['image'], out['mask']                # img: [H,W] 或 [H,W,1]

        img = img.astype(np.float32) * 2.0 - 1.0            # [-1,1]

        # ---- 转 tensor ----
        if img.ndim == 2:
            img = img[None, :, :]                           # [1,H,W]
        else:
            img = img.transpose(2, 0, 1)                    # [C,H,W]
        img = torch.from_numpy(np.ascontiguousarray(img)).float()
        
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)                     # 视图，省内存

        lab = torch.as_tensor(lab, dtype=torch.long)

        filename = os.path.splitext(os.path.basename(self.all_data_path[index]))[0]
        return {'image': img, 'label': lab, 'filename': filename}

    def __len__(self):
        return len(self.all_data_path)

class MMWHSDataset_2transform(data.Dataset):
    """Mean Teacher 数据集：每个样本返回 (weak, strong) 两个视图。

    weak  : 仅 Resize，供 teacher 生成稳定的伪标签
    strong: 完整增强，供 student 学习
    非训练阶段（val/test）weak 与 strong 都只做 Resize。
    """

    def __init__(self, rootdir: str, modality: str, phase: str = 'train',
                 mode: str = 'adapt'):
        self.rootdir = rootdir
        self.modality = modality
        self.phase = phase
        self.mode = mode

        self.all_data_path, self.all_label_path = collect_image_label_paths(
            rootdir, modality, phase)

        # weak: 只 Resize；strong: 训练集做增强，否则也只 Resize
        self.weak_transform = build_adapt_transforms(is_train=False)
        self.strong_transform = build_adapt_transforms(
            is_train=(phase == 'train')
        )

    def _to_tensor(self, img, lab):
        img = img.astype(np.float32) * 2.0 - 1.0            # [-1,1]

        if img.ndim == 2:
            img = img[None, :, :]                           # [1,H,W]
        else:
            img = img.transpose(2, 0, 1)                    # [C,H,W]
        img = torch.from_numpy(np.ascontiguousarray(img)).float()

        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)                     # 视图，省内存

        lab = torch.as_tensor(lab, dtype=torch.long)
        return img, lab

    def __getitem__(self, index):
        # ---- 读取 ----
        img = np.load(self.all_data_path[index])
        lab = np.load(self.all_label_path[index]).astype(np.int64)

        # ---- 统一成 [H,W]（单通道），避免 permute 维度坑 ----
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]

        # ---- 先到 [0,1]，再分别做弱/强增强，最后映射到 [-1,1] ----
        img = to_unit_range(img, self.modality)             # [0,1]

        out_w = self.weak_transform(image=img, mask=lab)
        img_w, lab_w = out_w['image'], out_w['mask']

        out_s = self.strong_transform(image=img, mask=lab)
        img_s, lab_s = out_s['image'], out_s['mask']

        img_w, lab_w = self._to_tensor(img_w, lab_w)
        img_s, lab_s = self._to_tensor(img_s, lab_s)

        filename = os.path.splitext(os.path.basename(self.all_data_path[index]))[0]
        sample_w = {'image': img_w, 'label': lab_w, 'filename': filename}
        sample_s = {'image': img_s, 'label': lab_s, 'filename': filename}
        return sample_w, sample_s

    def __len__(self):
        return len(self.all_data_path)