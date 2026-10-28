import torch.utils.data as data
import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
'''
AA : Ascending Aorta | 升主动脉 | 从左心室发出的大动脉起始段，向上走行。是体循环的主干血管。 
LV : Left Ventricle | 左心室 | 心脏左侧下方的腔室，负责将富氧血液泵入主动脉，供应全身。心肌最厚。
MYO : Myocardium | 心肌（左心室心肌壁） | 特指左心室的心肌组织（不包括心腔），即 LV 的肌肉壁。
RV : Right Ventricle | 右心室 | 心脏右侧下方的腔室，负责将静脉血泵入肺动脉，进行肺循环。

contour_map = {
    "bg": 0,
    "lv_myo": 1,
    "la_blood": 2,
    "lv_blood": 3,
    "aa": 4,
}
'''
import albumentations as A

train_transforms = A.Compose([
    # 1. 空间增强
    A.Rotate(limit=15, p=0.5),
    A.Affine(translate_percent=(-0.1, 0.1), scale=(0.9, 1.1), rotate=0, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    
    # 2. 外观增强 (应对 Domain Shift)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(std_range=(0.03, 0.07), p=0.3),
    # A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    
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
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
    # A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    
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

IMAGE_TAG = 'img'
LABEL_TAG = 'label'

def collect_image_label_paths(rootdir, modality, phase):
    data_dir = os.path.join(rootdir, f'{modality}_{phase}')
    image_paths = sorted(glob.glob(os.path.join(data_dir, f'*{IMAGE_TAG}*.npy')))
    label_paths = sorted(glob.glob(os.path.join(data_dir, f'*{LABEL_TAG}*.npy')))
    return image_paths, label_paths

class mmwhs_dataset(data.Dataset):
    def __init__(self, rootdir, modality, phase='train',mode = 'source'):
        self.rootdir = rootdir          # root_path
        self.modality = modality        # ct
        self.phase = phase              # train
        self.mode = mode
        self.all_data_path = []
        self.all_label_path = []

        self.all_data_path, self.all_label_path = collect_image_label_paths(
            self.rootdir, self.modality, phase)
        
    def __getitem__(self, index):
        img,lab = np.load(self.all_data_path[index]),np.load(self.all_label_path[index])
        # e.g. ct_train/ct_train_slice7_img.npy
        filename = os.path.basename(self.all_data_path[index]).split('.')[0]

        # image : [256 ,256, 3]
        # label : [256 ,256]

        if self.mode == 'source' and self.phase == 'train': 
            augmented = train_transforms(image=img, mask=lab)
            img = augmented['image']
            lab = augmented['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()    # [3,256,256]
        lab = torch.from_numpy(lab.astype(np.int64)).long()     # [256, 256]

        sample = {
            'image': img,
            'label': lab,
            'filename': filename
        }
        
        return sample
    
    def __len__(self):
        return len(self.all_data_path)


class mmwhs_dataset_normal(data.Dataset):
    def __init__(self, rootdir, modality, phase='train',mode = 'source'):
        self.rootdir = rootdir          # root_path
        self.modality = modality        # ct
        self.phase = phase              # train
        self.mode = mode
        self.all_data_path = []
        self.all_label_path = []

        self.all_data_path, self.all_label_path = collect_image_label_paths(
            self.rootdir, self.modality, phase)
        
    def __getitem__(self, index):
        img,lab = np.load(self.all_data_path[index]),np.load(self.all_label_path[index])

        denominator = img.max() - img.min()
        if denominator == 0:
            denominator = 1
        img = (img - img.min()) / denominator
        # e.g. ct_train/ct_train_slice7_img.npy
        filename = os.path.basename(self.all_data_path[index]).split('.')[0]

        # image : [256 ,256, 3]  values range : [0.0,1.0]
        # label : [256 ,256]

        if self.mode == 'source' and self.phase == 'train': 
            augmented = train_transforms(image=img, mask=lab)
            img = augmented['image']
            lab = augmented['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()    # [3,256,256]
        lab = torch.from_numpy(lab.astype(np.int64)).long()     # [256, 256]

        sample = {
            'image': img,
            'label': lab,
            'filename': filename
        }
        
        return sample
    
    def __len__(self):
        return len(self.all_data_path)
    

class mmwhs_dataset_2transform(data.Dataset):
    def __init__(self, rootdir, modality, phase='train',mode = 'source'):
        self.rootdir = rootdir          # root_path
        self.modality = modality        # ct
        self.phase = phase              # train
        self.mode = mode
        self.all_data_path = []
        self.all_label_path = []

        self.all_data_path, self.all_label_path = collect_image_label_paths(
            self.rootdir, self.modality, phase)
        
    def __getitem__(self, index):
        img,lab = np.load(self.all_data_path[index]),np.load(self.all_label_path[index])
        # e.g. ct_train/ct_train_slice7_img.npy
        filename = os.path.basename(self.all_data_path[index]).split('.')[0]

        # image : [256 ,256, 3]
        # label : [256 ,256]

        if self.phase == 'train': 
            augmented = adapt_transforms(image=img, mask=lab)
            img_aug = augmented['image']
            lab_aug = augmented['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()    # [3,256,256]
        img_aug = torch.from_numpy(img_aug).permute(2, 0, 1).float()    # [3,256,256]
        lab_aug = torch.from_numpy(lab_aug.astype(np.int64)).long()     # [256, 256]

        sample_w = {
            'image': img,
            'label': lab,
            'filename': filename
        }

        sample_s = {
            'image': img_aug,
            'label': lab_aug,
            'filename': filename
        }
        return sample_w, sample_s
    
    def __len__(self):
        return len(self.all_data_path)

class mmwhs_dataset_normal_2transform(data.Dataset):
    def __init__(self, rootdir, modality, phase='train',mode = 'source'):
        self.rootdir = rootdir          # root_path
        self.modality = modality        # ct
        self.phase = phase              # train
        self.mode = mode
        self.all_data_path = []
        self.all_label_path = []

        self.all_data_path, self.all_label_path = collect_image_label_paths(
            self.rootdir, self.modality, phase)
        
    def __getitem__(self, index):
        img,lab = np.load(self.all_data_path[index]),np.load(self.all_label_path[index])
        # e.g. ct_train/ct_train_slice7_img.npy
        denominator = img.max() - img.min()
        if denominator == 0:
            denominator = 1
        img = (img - img.min()) / denominator
        filename = os.path.basename(self.all_data_path[index]).split('.')[0]

        # image : [256 ,256, 3]
        # label : [256 ,256]

        if self.phase == 'train': 
            augmented = adapt_transforms(image=img, mask=lab)
            img_aug = augmented['image']
            lab_aug = augmented['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()    # [3,256,256]
        img_aug = torch.from_numpy(img_aug).permute(2, 0, 1).float()    # [3,256,256]
        lab_aug = torch.from_numpy(lab_aug.astype(np.int64)).long()     # [256, 256]

        sample_w = {
            'image': img,
            'label': lab,
            'filename': filename
        }

        sample_s = {
            'image': img_aug,
            'label': lab_aug,
            'filename': filename
        }
        return sample_w, sample_s
    
    def __len__(self):
        return len(self.all_data_path)
    