##load data
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import yaml
import random
import configparser
import math
from torch.nn import init
from scipy import ndimage
import SimpleITK as sitk 
import torch.nn as nn
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
def random_sample(LV):
    LV[LV == 1] = 255
    LV = LV.astype(np.uint8)
    white_indices_LV = np.where(LV == 255)
    points = []
    input_labels = []
    process_part_1(white_indices_LV, points, input_labels)
    if all(points_part is None for points_part in points):
        return None
    else:
        all_points = np.concatenate([points_part for points_part in points if points_part is not None])
        all_input_labels = np.concatenate([input_labels_part for input_labels_part in input_labels if input_labels_part is not None])
        return all_points, all_input_labels

def process_part_center(MYO, points, input_labels):
    contours, _ = cv2.findContours(MYO, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        points.append(None)
        input_labels.append(None)
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        contour = largest_contour
        M = cv2.moments(contour)
        if M['m00'] != 0:
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            center_point = (centroid_x, centroid_y)
            points.append(np.array([center_point]))
            input_labels.append(np.array([1]))
def process_part_center_0(MYO, points, input_labels):
    contours, _ = cv2.findContours(MYO, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        points.append(None)
        input_labels.append(None)
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        contour = largest_contour
        M = cv2.moments(contour)
        if M['m00'] != 0:
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            center_point = (centroid_x, centroid_y)
            points.append(np.array([center_point]))
            input_labels.append(np.array([0]))

def process_part_1(white_indices, points, input_labels):
    if white_indices[0].size < 3:
        points.append(None)
        input_labels.append(None)
    else:
        random_indices = np.random.choice(len(white_indices[0]), size=3, replace=False)
        random_points = [(white_indices[0][i], white_indices[1][i]) for i in random_indices]
        points.append(np.array(random_points))
        points[-1] = np.column_stack((points[-1][:, 1], points[-1][:, 0]))
        input_labels.append(np.array([1, 1, 1]))

def process_part_0(white_indices, points, input_labels):
    if white_indices[0].size < 3:
        points.append(None)
        input_labels.append(None)
    else:
        random_indices = np.random.choice(len(white_indices[0]), size=3, replace=False)
        random_points = [(white_indices[0][i], white_indices[1][i]) for i in random_indices]
        points.append(np.array(random_points))
        points[-1] = np.column_stack((points[-1][:, 1], points[-1][:, 0]))
        input_labels.append(np.array([0, 0, 0]))
def norm_0_255(x):

    x_shape = list(x.shape)
    if (len(x_shape) == 5):
        [N, C, D, H, W] = x_shape
        new_shape = [N * D, C, H, W]
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, new_shape)
    normalized_x = ((x + 1) / 2) * 255
    normalized_x = normalized_x.to(torch.uint8)
    image_array = normalized_x.cpu().detach().numpy()
    return image_array
def show_mask(mask, img, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.3])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image_bgr = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2BGRA) if mask_image.shape[2] == 4 else mask_image
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    combined_img = cv2.addWeighted(img_bgr, 1, mask_image_bgr, 0.7, 0)

    return combined_img
def show_mask_and_points(mask, points, labels, img, output, gt):
    marker_size = 2
    pos_points = points[labels == 1]
    neg_points = points[labels == 0]

    color_sam = np.array([30, 30, 150, 80])
    color_output = np.array([150, 30, 30, 80])
    color_gt = np.array([30 / 255, 150 / 255, 30 / 255, 80 / 255])

    h, w = mask.shape[-2:]
    mask_image_sam = mask.reshape(h, w, 1)
    mask_image_sam_1 = mask_image_sam.copy().astype(np.uint8)
    contours_sam, _ = cv2.findContours(mask_image_sam_1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mask_image_output = output.reshape(h, w, 1)
    mask_image_output_1 = mask_image_output.copy().astype(np.uint8)
    contours_output, _ = cv2.findContours(mask_image_output_1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mask_image_gt = gt.reshape(h, w, 1)
    mask_image_gt_1 = mask_image_gt.copy().astype(np.uint8)
    contours_gt, _ = cv2.findContours(mask_image_gt_1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_bgr = img

    combined_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)

    cv2.drawContours(combined_img, contours_sam, -1, (0, 0, 255, 255), thickness=1)
    cv2.drawContours(combined_img, contours_output, -1, (255, 0, 0, 255), thickness=1)
    cv2.drawContours(combined_img, contours_gt, -1, (0, 255, 0, 255), thickness=1)

    for pos_point in pos_points:

        cv2.circle(combined_img, pos_point, marker_size, color=(0, 255, 0, 255), thickness=-1)

    for neg_point in neg_points:

        cv2.circle(combined_img, neg_point, marker_size, color=(0, 0, 255, 255), thickness=-1)

    return combined_img

def show_points(coords, labels, img, marker_size=2):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    image = img.copy()
    image[image == 1] = 255
    # Convert image to BGR format if not already in that format
    img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()

    # Draw green and red points on the image
    for pos_point in pos_points:
        cv2.drawMarker(img_bgr, tuple(map(int, pos_point)), color=(0, 255, 0), markerType=cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=1)

    for neg_point in neg_points:
        cv2.drawMarker(img_bgr, tuple(map(int, neg_point)), color=(0, 0, 255), markerType=cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=1)

    return img_bgr

def show_points_GT(coords, labels, img, marker_size=2):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    image = img.copy()
    # Convert image to BGR format if not already in that format
    img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()

    # Draw green and red points on the image
    for pos_point in pos_points:
        cv2.drawMarker(img_bgr, tuple(map(int, pos_point)), color=(0, 255, 0), markerType=cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=1)

    for neg_point in neg_points:
        cv2.drawMarker(img_bgr, tuple(map(int, neg_point)), color=(0, 0, 255), markerType=cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=1)

    return img_bgr
def is_int(val_str):
    start_digit = 0
    if(val_str[0] =='-'):
        start_digit = 1
    flag = True
    for i in range(start_digit, len(val_str)):
        if(str(val_str[i]) < '0' or str(val_str[i]) > '9'):
            flag = False
            break
    return flag

def is_float(val_str):
    flag = False
    if('.' in val_str and len(val_str.split('.'))==2 and not('./' in val_str)):
        if(is_int(val_str.split('.')[0]) and is_int(val_str.split('.')[1])):
            flag = True
        else:
            flag = False
    elif('e' in val_str and val_str[0] != 'e' and len(val_str.split('e'))==2):
        if(is_int(val_str.split('e')[0]) and is_int(val_str.split('e')[1])):
            flag = True
        else:
            flag = False       
    else:
        flag = False
    return flag 

def is_bool(var_str):
    if( var_str.lower() =='true' or var_str.lower() == 'false'):
        return True
    else:
        return False
    
def parse_bool(var_str):
    if(var_str.lower() =='true'):
        return True
    else:
        return False
     
def is_list(val_str):
    if(val_str[0] == '[' and val_str[-1] == ']'):
        return True
    else:
        return False

def parse_list(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if(is_int(item)):
            output.append(int(item))
        elif(is_float(item)):
            output.append(float(item))
        elif(is_bool(item)):
            output.append(parse_bool(item))
        elif(item.lower() == 'none'):
            output.append(None)
        else:
            output.append(item)
    return output
    
def parse_value_from_string(val_str):
    if(is_int(val_str)):
        val = int(val_str)
    elif(is_float(val_str)):
        val = float(val_str)
    elif(is_list(val_str)):
        val = parse_list(val_str)
    elif(is_bool(val_str)):
        val = parse_bool(val_str)
    elif(val_str.lower() == 'none'):
        val = None
    else:
        val = val_str
    return val

def parse_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    output = {}
    for section in config.sections():
        output[section] = {}
        for key in config[section]:
            val_str = str(config[section][key])
            if(len(val_str)>0):
                val = parse_value_from_string(val_str)
                output[section][key] = val
            else:
                val = None
            print(section, key, val_str, val)
    return output


def load_npz(path):
    img = np.load(path)['arr_0']
    gt = np.load(path)['arr_1']
    return img, gt
    
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream,Loader=yaml.FullLoader)

def set_random(seed_id=3407):
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)   #for cpu
    torch.cuda.manual_seed_all(seed_id) #for GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# config setting
def is_int(val_str):
    start_digit = 0
    if(val_str[0] =='-'):
        start_digit = 1
    flag = True
    for i in range(start_digit, len(val_str)):
        if(str(val_str[i]) < '0' or str(val_str[i]) > '9'):
            flag = False
            break
    return flag

def is_float(val_str):
    flag = False
    if('.' in val_str and len(val_str.split('.'))==2 and not('./' in val_str)):
        if(is_int(val_str.split('.')[0]) and is_int(val_str.split('.')[1])):
            flag = True
        else:
            flag = False
    elif('e' in val_str and val_str[0] != 'e' and len(val_str.split('e'))==2):
        if(is_int(val_str.split('e')[0]) and is_int(val_str.split('e')[1])):
            flag = True
        else:
            flag = False       
    else:
        flag = False
    return flag 

def is_bool(var_str):
    if( var_str.lower() =='true' or var_str.lower() == 'false'):
        return True
    else:
        return False
    
def parse_bool(var_str):
    if(var_str.lower() =='true'):
        return True
    else:
        return False
     
def is_list(val_str):
    if(val_str[0] == '[' and val_str[-1] == ']'):
        return True
    else:
        return False

def parse_list(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if(is_int(item)):
            output.append(int(item))
        elif(is_float(item)):
            output.append(float(item))
        elif(is_bool(item)):
            output.append(parse_bool(item))
        elif(item.lower() == 'none'):
            output.append(None)
        else:
            output.append(item)
    return output

def parse_value_from_string(val_str):
    if(is_int(val_str)):
        val = int(val_str)
    elif(is_float(val_str)):
        val = float(val_str)
    elif(is_list(val_str)):
        val = parse_list(val_str)
    elif(is_bool(val_str)):
        val = parse_bool(val_str)
    elif(val_str.lower() == 'none'):
        val = None
    else:
        val = val_str
    return val

def parse_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    output = {}
    for section in config.sections():
        output[section] = {}
        for key in config[section]:
            val_str = str(config[section][key])
            if(len(val_str)>0):
                val = parse_value_from_string(val_str)
                output[section][key] = val
            else:
                val = None
            print(section, key, val_str, val)
    return output

class UnpairedDataset(Dataset):
    #get unpaired dataset, such as MR-CT dataset
    def __init__(self,A_path,B_path):
        listA = os.listdir(A_path)
        listB = os.listdir(B_path)
        self.listA = [os.path.join(A_path,k) for k in listA]
        self.listB = [os.path.join(B_path,k) for k in listB]
        self.Asize = len(self.listA)
        self.Bsize = len(self.listB)
        self.dataset_size = max(self.Asize,self.Bsize)
        
    def __getitem__(self,index):
        if self.Asize == self.dataset_size:
            A,A_gt = load_npz(self.listA[index])
            B,B_gt = load_npz(self.listB[random.randint(0, self.Bsize - 1)])
        else :
            B,B_gt = load_npz(self.listB[index])
            A,A_gt = load_npz(self.listA[random.randint(0, self.Asize - 1)])


        A = torch.from_numpy(A.copy()).unsqueeze(0).float()
        A_gt = torch.from_numpy(A_gt.copy()).unsqueeze(0).float()
        B = torch.from_numpy(B.copy()).unsqueeze(0).float()
        B_gt = torch.from_numpy(B_gt.copy()).unsqueeze(0).float()
        return A,A_gt,B,B_gt
        
    def __len__(self):
        return self.dataset_size

def crop_depth(img,lab,phase = 'train'):
    D,H,W = img.shape
    if D > 10:
        if phase == 'train':
            # target_ssh = np.random.randint(0, int(D-10), 1)[0]
            # zero_img = img[target_ssh:target_ssh+10,:,:]
            # zero_lab = lab[target_ssh:target_ssh+10,:,:]
            target_ssh = int((D-10)/2)
            zero_img = img[target_ssh:target_ssh+10,:,:]
            zero_lab = lab[target_ssh:target_ssh+10,:,:]
        elif phase == 'valid':
            zero_img,zero_lab = img,lab
        elif phase == 'feta':
            sample_indices = np.random.choice(D, size=10, replace=False)
            zero_img = np.zeros((10,H,W))
            zero_lab = np.zeros((10,H,W))
            for i, index in enumerate(sample_indices):
                zero_img[i] = img[index]
                zero_lab[i] = lab[index]
    else:
        zero_img = np.zeros((10,H,W))
        zero_lab = np.zeros((10,H,W))
        zero_img[0:D,:,:] = img
        zero_lab[0:D,:,:] = lab
    return zero_img,zero_lab

def winadj_mri(array):
    v0 = np.percentile(array, 1)
    v1 = np.percentile(array, 99)
    array[array < v0] = v0    
    array[array > v1] = v1  
    v0 = array.min() 
    v1 = array.max() 
    array = (array - v0) / (v1 - v0) * 2.0 - 1.0
    return array

def resize(img,lab):
    D,H,W = img.shape
    zoom = [1,256/H,256/W]
    img=ndimage.zoom(img,zoom,order=2)
    lab=ndimage.zoom(lab,zoom,order=0)
    return img,lab

class niiDataset(Dataset):
    def __init__(self, source_img,source_lab,dataset,target, phase = 'test'):
        self.dataset = dataset
        self.source_img = source_img
        self.source_lab = source_lab
        self.phase = phase
        nii_names = os.listdir(source_img)
        self.all_files = []
        for nii_name in nii_names:
            self.img_path = os.path.join(self.source_img, nii_name)
            if self.dataset == 'mms':
                self.lab_path = os.path.join(self.source_lab, nii_name[:-7] + '_gt.nii.gz')
            else:
                print(self.dataset)
                raise Exception('Unrecognized dataset.')
            
            self.nii_name = str(nii_name)
            self.all_files.append({
                "img": self.img_path,
                "lab": self.lab_path,
                "img_name": self.nii_name
            })    
    def __getitem__(self, index):
        fname = self.all_files[index]
        img_obj = sitk.ReadImage(fname["img"])
        A = sitk.GetArrayFromImage(img_obj) /1 
        lab_obj = sitk.ReadImage(fname["lab"])
        A_gt = sitk.GetArrayFromImage(lab_obj)
        if self.phase == 'train' and self.dataset == 'mms':
            A,A_gt = crop_depth(A,A_gt)
        A = winadj_mri(A)
        A,A_gt = resize(A,A_gt)
        A = torch.from_numpy(A.copy()).unsqueeze(0).float()
        A_gt = torch.from_numpy(A_gt.copy()).unsqueeze(0).float()
        return A.repeat(3,1,1,1), A_gt, fname["img_name"],fname["lab"]
           
    def __len__(self):
        return len(self.all_files)

def _aug_gaussian_noise_volume(img, sigma_range=(0.01, 0.03), p=0.5):
    """
    温和的高斯噪声。取代原 salt & pepper。
    img 已在 [-1, 1]，sigma 0.01~0.03 相当于 SNR 约 30~40dB，
    足以模拟扫描仪噪声，又不会破坏解剖结构。
    """
    if random.random() > p:
        return img
    sigma = random.uniform(*sigma_range)
    img = img.copy()
    noise = np.random.randn(*img.shape).astype(img.dtype) * sigma
    return np.clip(img + noise, -1.0, 1.0)


def _aug_adjust_light_volume(img,gamma_range=(0.9, 1.1), p=0.5):
    """
    温和的 gamma 抖动。取代原 [0.5, 3.5] 的极端分布。
    gamma=0.9 轻微变亮，gamma=1.1 轻微变暗，对比度变化 ±10% 左右。
    这足以提供多样性，又不会改变心肌/血池的相对对比度。
    """
    if random.random() > p:
        return img
    gamma = random.uniform(*gamma_range)
    invGamma = 1.0 / gamma
    img = img.copy()
    return np.sign(img) * (np.abs(img) ** invGamma)


def _aug_eraser_volume(img, s_l=0.1, s_h=0.3, r_1=0.3, r_2=0.6):
    if random.random() > 0.5:
        return img
    D, H, W = img.shape
    while True:
        s = np.random.uniform(s_l, s_h) * H * W
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, W)
        top = np.random.randint(0, H)
        if left + w <= W and top + h <= H:
            break
    c = np.random.uniform(-1, 1)
    img = img.copy()
    img[:, top:top + h, left:left + w] = c
    return img

def _apply_geometry_aug(img, label, flip_axes=(1,2), rot90_prob=0.5):
    """
    随机翻转 + 90度倍数旋转。
    Args:
        img: numpy array (D,H,W)
        label: numpy array (D,H,W)
        flip_axes: 允许翻转的轴元组，默认 (1,2) 表示沿 H 和 W 翻转。
        rot90_prob: 进行90度旋转的概率。
    """
    img = img.copy()
    label = label.copy()

    # 随机翻转
    for ax in flip_axes:
        if random.random() < 0.5:
            img = np.flip(img, axis=ax)
            label = np.flip(label, axis=ax)

    # 90度倍数旋转（在 H-W 平面，即 axes=(1,2)）
    if random.random() < rot90_prob:
        k = random.randint(0, 3)  # 0,1,2,3 对应 0°,90°,180°,270°
        if k != 0:
            img = np.rot90(img, k, axes=(1, 2))
            label = np.rot90(label, k, axes=(1, 2))

    return img, label

def strong_aug(img, mode='source', label=None):
    """
    强增强入口。
    Args:
        img: numpy array, shape (D, H, W)
        mode: 'source' 或 'adapt'
        label: 若提供且 mode='source'，则同步进行几何变换；否则忽略。
    Returns:
        (img_aug, label_aug) 如果 label 不为 None；否则只返回 img_aug。
    """
    img = img.copy()
    label_out = None

    # 几何变换（仅源域模式，且 label 存在时同步）
    if mode == 'source' and label is not None:
        img, label_out = _apply_geometry_aug(img, label)
    elif mode == 'source' and label is None:
        # 无标签时也可进行几何变换，但不推荐；此处仅对图像操作
        img, _ = _apply_geometry_aug(img, img.copy())

    # 强度增强（两种模式都执行）
    img = _aug_gaussian_noise_volume(img)
    img = _aug_adjust_light_volume(img)
    img = _aug_eraser_volume(img)

    if label is not None:
        return img, label_out
    else:
        return img

def strong_aug_woErase(img, mode='source', label=None):
    """
    强增强入口。
    Args:
        img: numpy array, shape (D, H, W)
        mode: 'source' 或 'adapt'
        label: 若提供且 mode='source'，则同步进行几何变换；否则忽略。
    Returns:
        (img_aug, label_aug) 如果 label 不为 None；否则只返回 img_aug。
        adapt 阶段不提供 erase 操作
    """
    img = img.copy()
    label_out = None

    # 几何变换（仅源域模式，且 label 存在时同步）
    if mode == 'source' and label is not None:
        img, label_out = _apply_geometry_aug(img, label)
    elif mode == 'source' and label is None:
        # 无标签时也可进行几何变换，但不推荐；此处仅对图像操作
        img, _ = _apply_geometry_aug(img, img.copy())

    # 强度增强（两种模式都执行）
    img = _aug_gaussian_noise_volume(img)
    img = _aug_adjust_light_volume(img)
    # img = _aug_eraser_volume(img)

    if label is not None:
        return img, label_out
    else:
        return img
    
class niiDataset_wAug(Dataset):
    def __init__(self, source_img, source_lab, dataset, phase='train',mode = 'source'):
        self.dataset = dataset
        self.source_img = source_img
        self.source_lab = source_lab
        self.phase = phase
        self.mode = mode
        nii_names = os.listdir(source_img)
        self.all_files = []
        for nii_name in nii_names:
            self.img_path = os.path.join(self.source_img, nii_name)
            if self.dataset == 'mms':
                self.lab_path = os.path.join(self.source_lab, nii_name[:-7] + '_gt.nii.gz')
            else:
                raise Exception('Unrecognized dataset.')
            self.nii_name = str(nii_name)
            self.all_files.append({
                "img": self.img_path,
                "lab": self.lab_path,
                "img_name": self.nii_name
            })

    def __getitem__(self, index):
        fname = self.all_files[index]
        img_obj = sitk.ReadImage(fname["img"])
        A = sitk.GetArrayFromImage(img_obj)
        lab_obj = sitk.ReadImage(fname["lab"])
        A_gt = sitk.GetArrayFromImage(lab_obj)
        if self.phase == 'train' and self.dataset == 'mms':
            A, A_gt = crop_depth(A, A_gt)
        A = winadj_mri(A)
        A, A_gt = resize(A, A_gt)
      
        if self.mode == 'source':
            # 源域：可包含几何变换，标签需同步
            A_s, A_gt_s = strong_aug(A, mode='source', label=A_gt)
        else:  # adapt 模式
            A_s = strong_aug(A, mode='adapt')
            A_gt_s = A_gt  # 标签不变

        A_s = torch.from_numpy(A_s.copy()).unsqueeze(0).float()
        A_gt_s = torch.from_numpy(A_gt_s.copy()).unsqueeze(0).float()

        return A_s.repeat(3, 1, 1, 1), A_gt_s, fname["img_name"]

    def __len__(self):
        return len(self.all_files)
    

class niiDataset_2transform_v2(Dataset):
    def __init__(self, source_img, source_lab, dataset, phase='train',mode = 'source'):
        self.dataset = dataset
        self.source_img = source_img
        self.source_lab = source_lab
        self.phase = phase
        self.mode = mode
        nii_names = os.listdir(source_img)
        self.all_files = []
        for nii_name in nii_names:
            self.img_path = os.path.join(self.source_img, nii_name)
            if self.dataset == 'mms':
                self.lab_path = os.path.join(self.source_lab, nii_name[:-7] + '_gt.nii.gz')
            else:
                raise Exception('Unrecognized dataset.')
            self.nii_name = str(nii_name)
            self.all_files.append({
                "img": self.img_path,
                "lab": self.lab_path,
                "img_name": self.nii_name
            })

    def __getitem__(self, index):
        fname = self.all_files[index]
        img_obj = sitk.ReadImage(fname["img"])
        A = sitk.GetArrayFromImage(img_obj)
        lab_obj = sitk.ReadImage(fname["lab"])
        A_gt = sitk.GetArrayFromImage(lab_obj)
        if self.phase == 'train' and self.dataset == 'mms':
            A, A_gt = crop_depth(A, A_gt)
        A = winadj_mri(A)
        A, A_gt = resize(A, A_gt)

        A_w = A.copy()
        A_gt_w = A_gt.copy()
        
        if self.mode == 'source':
            # 源域：可包含几何变换，标签需同步
            A_s, A_gt_s = strong_aug(A, mode='source', label=A_gt)
        else:  # adapt 模式
            A_s = strong_aug(A, mode='adapt')
            A_gt_s = A_gt  # 标签不变
       
        A_w = torch.from_numpy(A_w.copy()).unsqueeze(0).float()
        A_s = torch.from_numpy(A_s.copy()).unsqueeze(0).float()
        A_gt_w = torch.from_numpy(A_gt.copy()).unsqueeze(0).float()
        A_gt_s = torch.from_numpy(A_gt_s.copy()).unsqueeze(0).float()

        sample_w = {'image': A_w.repeat(3, 1, 1, 1), 'label': A_gt_w, 'img_name': fname["img_name"]}
        sample_s = {'image': A_s.repeat(3, 1, 1, 1), 'label': A_gt_s, 'img_name': fname["img_name"]}
        return sample_w, sample_s

    def __len__(self):
        return len(self.all_files)
    
class niiDataset_2transform_woErase(Dataset):
    def __init__(self, source_img, source_lab, dataset, phase='train',mode = 'source'):
        self.dataset = dataset
        self.source_img = source_img
        self.source_lab = source_lab
        self.phase = phase
        self.mode = mode
        nii_names = os.listdir(source_img)
        self.all_files = []
        for nii_name in nii_names:
            self.img_path = os.path.join(self.source_img, nii_name)
            if self.dataset == 'mms':
                self.lab_path = os.path.join(self.source_lab, nii_name[:-7] + '_gt.nii.gz')
            else:
                raise Exception('Unrecognized dataset.')
            self.nii_name = str(nii_name)
            self.all_files.append({
                "img": self.img_path,
                "lab": self.lab_path,
                "img_name": self.nii_name
            })

    def __getitem__(self, index):
        fname = self.all_files[index]
        img_obj = sitk.ReadImage(fname["img"])
        A = sitk.GetArrayFromImage(img_obj)
        lab_obj = sitk.ReadImage(fname["lab"])
        A_gt = sitk.GetArrayFromImage(lab_obj)
        if self.phase == 'train' and self.dataset == 'mms':
            A, A_gt = crop_depth(A, A_gt)
        A = winadj_mri(A)
        A, A_gt = resize(A, A_gt)

        A_w = A.copy()
        A_gt_w = A_gt.copy()
        
        if self.mode == 'source':
            # 源域：可包含几何变换，标签需同步
            A_s, A_gt_s = strong_aug_woErase(A, mode='source', label=A_gt)
        else:  # adapt 模式
            A_s = strong_aug_woErase(A, mode='adapt')
            A_gt_s = A_gt  # 标签不变
       
        A_w = torch.from_numpy(A_w.copy()).unsqueeze(0).float()
        A_s = torch.from_numpy(A_s.copy()).unsqueeze(0).float()
        A_gt_w = torch.from_numpy(A_gt.copy()).unsqueeze(0).float()
        A_gt_s = torch.from_numpy(A_gt_s.copy()).unsqueeze(0).float()

        sample_w = {'image': A_w.repeat(3, 1, 1, 1), 'label': A_gt_w, 'img_name': fname["img_name"]}
        sample_s = {'image': A_s.repeat(3, 1, 1, 1), 'label': A_gt_s, 'img_name': fname["img_name"]}
        return sample_w, sample_s

    def __len__(self):
        return len(self.all_files)

import os
import random
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class niiDataset_2transform_split(Dataset):
    def __init__(self, source_img, source_lab, dataset,
                 phase='train', mode='source',patient_list = None,
                 ratios=(0.7, 0.1, 0.2), seed=42):
        self.dataset = dataset
        self.source_img = source_img
        self.source_lab = source_lab
        self.phase = phase          # 'train' / 'val' / 'test' / 'all'
        self.mode = mode            # 'source' / 'target'
        self.ratios = ratios
        self.seed = seed

        # 排序，保证三个 split 实例化时文件顺序一致
        nii_names = sorted(os.listdir(source_img))

        all_files = []
        for nii_name in nii_names:
            img_path = os.path.join(self.source_img, nii_name)
            if self.dataset == 'mms':
                lab_path = os.path.join(self.source_lab, nii_name[:-7] + '_gt.nii.gz')
            else:
                raise Exception('Unrecognized dataset.')

            all_files.append({
                "img": img_path,
                "lab": lab_path,
                "img_name": nii_name
            })

        # ---------- ★ 按患者 ID 分组 ----------
        if patient_list is not None:
            sel_pids = set(patient_list)
        else:
            from collections import defaultdict
            patient2files = defaultdict(list)
            for nii_name in nii_names:
                pid = nii_name.split('_')[0]              # 'A1D0Q7'
                patient2files[pid].append(nii_name)

            patient_ids = sorted(patient2files.keys())   
            print(f'患者共有 : {patient_ids}')
            # ---------- 按 volume 划分 7:1:2 ----------
            rng = random.Random(seed)
            rng.shuffle(patient_ids)

            n = len(patient_ids)
            n_train = int(n * ratios[0])
            n_val = int(n * ratios[1])
            # 剩余全部给 test，避免取整后漏掉
            n_test = n - n_train - n_val

            if self.phase == 'train':
                sel_pids = set(patient_ids[:n_train])
            elif self.phase == 'val':
                sel_pids = set(patient_ids[n_train:n_train + n_val])
            elif self.phase == 'test':
                sel_pids = set(patient_ids[n_train + n_val:])
            elif self.phase == 'all':
                sel_pids = set(patient_ids)
            else:
                raise ValueError(f'Unknown split: {self.phase}')

        # ★ 关键：给 self.all_files 赋值
        self.all_files = [f for f in all_files
                        if f['img_name'].split('_')[0] in sel_pids]

        # 安全检查：三个 split 的患者绝不重叠
        self.sel_pids = sel_pids
        print(f'[niiDataset_2transform_split] phase={phase}, '
              f'#selected patients={len(sel_pids)}, '
              f'#selected files={len(self.all_files)}')
        print(f'[niiDataset_2transform_split] patients: {sorted(sel_pids)}')

    def __getitem__(self, index):
        fname = self.all_files[index]

        img_obj = sitk.ReadImage(fname["img"])
        A = sitk.GetArrayFromImage(img_obj)

        lab_obj = sitk.ReadImage(fname["lab"])
        A_gt = sitk.GetArrayFromImage(lab_obj)

        if self.phase == 'train' and self.dataset == 'mms':
            A, A_gt = crop_depth(A, A_gt)

        A = winadj_mri(A)
        A, A_gt = resize(A, A_gt)

        # ================= 训练：返回两份（弱增强 + 强增强）=================
        if self.phase == 'train':
            A_w = A.copy()
            A_gt_w = A_gt.copy()

            if self.mode == 'source':
                A_s, A_gt_s = strong_aug_woErase(A, mode='source', label=A_gt)
            else:  # adapt 模式：只对图像做增强，标签不变
                A_s = strong_aug_woErase(A, mode='adapt')
                A_gt_s = A_gt

            A_w    = torch.from_numpy(A_w.copy()).unsqueeze(0).float()
            A_s    = torch.from_numpy(A_s.copy()).unsqueeze(0).float()
            A_gt_w = torch.from_numpy(A_gt_w.copy()).unsqueeze(0).float()
            A_gt_s = torch.from_numpy(A_gt_s.copy()).unsqueeze(0).float()

            sample_w = {
                'image': A_w.repeat(3, 1, 1, 1),
                'label': A_gt_w,
                'img_name': fname["img_name"]
            }
            sample_s = {
                'image': A_s.repeat(3, 1, 1, 1),
                'label': A_gt_s,
                'img_name': fname["img_name"]
            }
            return sample_w, sample_s

        # ================= 验证 / 测试：只返回一份 =================
        else:
            A_w = torch.from_numpy(A.copy()).unsqueeze(0).float()
            A_gt_w = torch.from_numpy(A_gt.copy()).unsqueeze(0).float()

            sample = {
                'image': A_w.repeat(3, 1, 1, 1),
                'label': A_gt_w,
                'img_name': fname["img_name"]
            }
            return sample

    def __len__(self):
        return len(self.all_files)