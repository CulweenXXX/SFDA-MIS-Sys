import torch.utils.data as data
import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import random,math
from skimage.exposure import match_histograms
'''
AA : Ascending Aorta | 升主动脉 | 从左心室发出的大动脉起始段，向上走行。是体循环的主干血管。 
LV : Left Ventricle | 左心室 | 心脏左侧下方的腔室，负责将富氧血液泵入主动脉，供应全身。心肌最厚。
MYO : Myocardium | 心肌（左心室心肌壁） | 特指左心室的心肌组织（不包括心腔），即 LV 的肌肉壁。
RV : Right Ventricle | 右心室 | 心脏右侧下方的腔室，负责将静脉血泵入肺动脉，进行肺循环。
'''
import albumentations as A

import cv2
import albumentations as A

train_transforms = A.Compose([
    # 1. 空间增强
    A.Rotate(limit=15, p=0.5),
    A.Affine(translate_percent=(-0.1, 0.1), scale=(0.9, 1.1), rotate=0, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    
    # 2. 外观增强 (应对 Domain Shift)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(std_range=(0.03, 0.07), p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    
    # 3. 防止过拟合的遮挡
    A.CoarseDropout(
        num_holes_range=(1, 8),
        hole_height_range=(16, 16),
        hole_width_range=(16, 16),
        fill=0,
        fill_mask=0,
        p=0.3
    ),
    
    # 4. 统一尺寸
    A.Resize(height=256, width=256)
])

adapt_transforms = A.Compose([
    # 1. 空间增强
    # A.Rotate(limit=15, p=0.5),
    # A.Affine(translate_percent=(-0.1, 0.1), scale=(0.9, 1.1), rotate=0, p=0.5),
    # A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    
    # 2. 外观增强 (应对 Domain Shift)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(std_range=(0.03, 0.07), p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    
    # 3. 防止过拟合的遮挡
    # A.CoarseDropout(
    #     num_holes_range=(1, 8),
    #     hole_height_range=(16, 16),
    #     hole_width_range=(16, 16),
    #     fill=0,
    #     fill_mask=0,
    #     p=0.3
    # ),
    
    # 4. 统一尺寸
    A.Resize(height=256, width=256)
])

IMAGE_SUBDIR = 'images'
LABEL_SUBDIR = 'labels'
IMAGE_TAG = '_img_'
LABEL_TAG = '_lab_'

def parse_case_and_slice(file_path):
    """从 ``{modality}_{split}_{case}_img_{slice}.npy`` 解析病例名与切片序号。"""
    '''
        os.path.basename(file_path) : 取路径的最后一段，包含扩展名
        os.path.splitext() : 把文件名拆成两部分 (root,ext)
    '''
    stem = os.path.splitext(os.path.basename(file_path))[0]
    parts = stem.split('_')
    if len(parts) >= 3 and parts[-1].isdigit():
        return '_'.join(parts[:-2]), int(parts[-1])
    return stem, -1


def image_to_label_path(image_path):
    image_dir = os.path.dirname(image_path)
    label_dir = os.path.join(os.path.dirname(image_dir), LABEL_SUBDIR)
    label_path = os.path.join(
        label_dir,
        os.path.basename(image_path).replace(IMAGE_TAG, LABEL_TAG),
    )
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f'label file not found for {image_path}: {label_path}')
    return label_path


def load_image_and_label(image_path, label_path=None):
    if label_path is None:
        label_path = image_to_label_path(image_path)

    img = np.load(image_path)
    if img.ndim == 2:
        img = img[None, ...]
    elif img.ndim == 3 and img.shape[-1] == 1 and img.shape[0] != 1:
        img = img.transpose(2, 0, 1)
    elif img.ndim == 3 and img.shape[0] == 1:
        pass
    else:
        raise ValueError(f'unexpected image shape {img.shape} for {image_path}')
    img = img.astype(np.float32)

    label = np.squeeze(np.load(label_path))
    if label.ndim != 2:
        raise ValueError(f'unexpected label shape {label.shape} for {label_path}')
    label = label.astype(np.uint8)

    return img, label, label_path


def _prepare_slice(image_path, label_path=None):
    """读取单张切片并缩放到 256x256。"""
    img, seg, _ = load_image_and_label(image_path, label_path)

    img = torch.from_numpy(img).float().unsqueeze(0)            # [1,1,H,W]
    img = F.interpolate(
        img, size=(256, 256),
        mode='bilinear', align_corners=False,
    )[0]                                                         # [1,256,256]

    seg = torch.from_numpy(seg).float().unsqueeze(0).unsqueeze(0)
    seg = F.interpolate(seg, size=(256, 256), mode='nearest')[0, 0].long()

    img = np.repeat(img.numpy(), 3, axis=0).transpose((1, 2, 0))  # [256,256,3]
    return np.ascontiguousarray(img), seg.numpy()


def collect_image_label_paths(rootdir, modality, phase):
    data_dir = os.path.join(rootdir, f'{modality}_{phase}')
    image_dir = os.path.join(data_dir, IMAGE_SUBDIR)
    image_paths = sorted(glob.glob(os.path.join(image_dir, f'*{IMAGE_TAG}*.npy')))
    if not image_paths:
        raise FileNotFoundError(f'no image files found in {image_dir}')
    label_paths = [image_to_label_path(path) for path in image_paths]
    return image_paths, label_paths

class mmwhs_dataset(data.Dataset):
    def __init__(self, rootdir, modality, phase='train',mode = 'source'):
        self.rootdir = rootdir
        self.modality = modality
        self.phase = phase
        self.mode = mode
        self.all_data_path = []
        self.all_label_path = []
        self.all_case_names = []
        self.all_slice_idx = []

        self.all_data_path, self.all_label_path = collect_image_label_paths(
            self.rootdir, self.modality, phase)
        for path in self.all_data_path:
            case_name, slice_idx = parse_case_and_slice(path)
            self.all_case_names.append(case_name)
            self.all_slice_idx.append(slice_idx)


    def __getitem__(self, index):
        '''
            image is [256,256] float32 in [0,1]
            label is [256,256] uint8 with values 0~4
        '''
        name = self.all_data_path[index]
        case_name = self.all_case_names[index]
        slice_idx = self.all_slice_idx[index]

        ## 1. loading image and preprocessing
        img, seg = _prepare_slice(name, self.all_label_path[index])
        # image : [256 ,256, 3]
        # label : [256 ,256]
        # 只增强训练阶段
        if self.mode == 'source' and self.phase == 'train': 
            augmented = train_transforms(image=img, mask=seg)
            img = augmented['image']
            seg = augmented['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()    # [3,256,256]

        seg = np.squeeze(seg)                                   # 删除单通道维度
        seg = torch.from_numpy(seg.astype(np.int64)).long()     # [256, 256]

        sample = {
            'image': img,
            'label': seg,
            'img_name': name,
            'case_name': case_name,
            'slice_idx': slice_idx,
        }
        
        return sample
    
    def __len__(self):
        return len(self.all_data_path)


class mmwhs_dataset_2transform(data.Dataset):
    def __init__(self, rootdir, modality, phase='train',mode = 'source'):
        self.rootdir = rootdir
        self.modality = modality
        self.phase = phase
        self.mode = mode
        self.all_data_path = []
        self.all_label_path = []
        self.all_case_names = []
        self.all_slice_idx = []

        self.all_data_path, self.all_label_path = collect_image_label_paths(
            self.rootdir, self.modality, phase)
        for path in self.all_data_path:
            case_name, slice_idx = parse_case_and_slice(path)
            self.all_case_names.append(case_name)
            self.all_slice_idx.append(slice_idx)


    def __getitem__(self, index):
        '''
            image is [256,256] float32 in [0,1]
            label is [256,256] uint8 with values 0~4
        '''
        name = self.all_data_path[index]
        case_name = self.all_case_names[index]
        slice_idx = self.all_slice_idx[index]

        ## 1. loading image and preprocessing
        img, seg = _prepare_slice(name, self.all_label_path[index])
        # image : [256 ,256, 3]
        # label : [256 ,256]
        # 只增强训练阶段
        if self.phase == 'train': 
            augmented = adapt_transforms(image=img, mask=seg)
            img_aug = augmented['image']
            seg_aug = augmented['mask']
        else:
            img_aug,seg_aug = img,seg
            
        img = torch.from_numpy(img).permute(2, 0, 1).float()                # [3,256,256]
        img_aug = torch.from_numpy(img_aug).permute(2, 0, 1).float()
        # seg_aug = np.squeeze(seg_aug)                                     # 删除单通道维度
        seg_aug = torch.from_numpy(seg_aug.astype(np.int64)).long()         # [256, 256]

        sample_w = {
            'image': img,
            'label': seg,
            'img_name': name,
            'case_name': case_name,
            'slice_idx': slice_idx,
        }

        sample_s = {
            'image': img_aug,
            'label': seg_aug,
            'img_name': name,
            'case_name': case_name,
            'slice_idx': slice_idx,
        }
        
        return sample_w,sample_s
    
    def __len__(self):
        return len(self.all_data_path)