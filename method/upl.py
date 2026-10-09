import argparse
from torch import Tensor
from utils.metrics import *
from utils.tools import *

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='0')

parser.add_argument('--model-file', type=str, default='logs_train/mms/checkpoint_80.pth.tar')  # source weight path

parser.add_argument('--model', type=str, default='Deeplab', help='Deeplab')
parser.add_argument('--out-stride', type=int, default=16)
parser.add_argument('--sync-bn', type=bool, default=True)
parser.add_argument('--freeze-bn', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr-decrease-rate', type=float, default=0.95, help='ratio multiplied to initial lr')
parser.add_argument('--lr-decrease-epoch', type=int, default=1, help='interval epoch number for lr decrease')

parser.add_argument('--data-dir', default='/data0/grchen/Data/M&MS')
parser.add_argument('--dataset', type=str, default='mms')
parser.add_argument('--target', type=str, default='B', help='target domain (B/C/D)')
parser.add_argument('--split', type=str, default='train', choices=['train', 'valid', 'test'],
                    help='which split to evaluate on (target domains only have labeled data under train/)')
parser.add_argument('--batch-size', type=int, default=2)

parser.add_argument('--model-ema-rate', type=float, default=0.98)
parser.add_argument('--pseudo-label-threshold', type=float, default=0.75)
parser.add_argument('--mean-loss-calc-bound-ratio', type=float, default=0.2)
parser.add_argument('--note', type=str, default='descript')
parser.add_argument('--interval-valid',type=int,default=1)
args = parser.parse_args()

import os

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import os.path as osp

import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
torch.backends.cudnn.enabled = False
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloaders import mms_dataloader
from datetime import datetime
import networks.deeplabv3 as netd
import torch.backends.cudnn as cudnn
import random
import sys
import copy

seed = 42
savefig = False
get_hd = True
model_save = True
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

NUM_CLASSES = 4
NAMES = ['BG','LV','MYO','RV']


class UPLDeepLabV3Plus(nn.Module):
    def __init__(self, source_model, num_decoders=4):
        super().__init__()
        '''
            model has been loaded state dict
        '''
        self.backbone = source_model.backbone
        self.aspp = source_model.aspp

        self.aux_dec1 = copy.deepcopy(source_model.decoder)
        self.aux_dec2 = copy.deepcopy(source_model.decoder)
        self.aux_dec3 = copy.deepcopy(source_model.decoder)
        self.aux_dec4 = copy.deepcopy(source_model.decoder)

        self.pl_threshold = 0.95
        
    def forward(self, x):
        A_1 = rotate_single_with_label(x, 1)
        A_2 = rotate_single_with_label(x, 2)
        A_3 = rotate_single_with_label(x, 3)
        A_4 = x

        low_1, high_1 = self.backbone(A_1, return_features=True)
        low_2, high_2 = self.backbone(A_2, return_features=True)
        low_3, high_3 = self.backbone(A_3, return_features=True)
        low_4, high_4 = self.backbone(A_4, return_features=True)

        aspp_1 = self.aspp(high_1)
        aspp_2 = self.aspp(high_2)
        aspp_3 = self.aspp(high_3)
        aspp_4 = self.aspp(high_4)

        self.aux_seg_1 = self.aux_dec1(low_1, aspp_1).softmax(1)
        self.aux_seg_2 = self.aux_dec2(low_2, aspp_2).softmax(1)
        self.aux_seg_3 = self.aux_dec3(low_3, aspp_3).softmax(1)
        self.aux_seg_4 = self.aux_dec4(low_4, aspp_4).softmax(1)

        self.aux_seg_1 = rotate_single_with_label(self.aux_seg_1, 3)
        self.aux_seg_2 = rotate_single_with_label(self.aux_seg_2, 2)
        self.aux_seg_3 = rotate_single_with_label(self.aux_seg_3, 1)

        return (self.aux_seg_1 + self.aux_seg_2 + self.aux_seg_3 + self.aux_seg_4) / 4.0

    def save_nii(self,x):
        self.forward(x)
        pred_aux1 = self.aux_seg_1.cpu().detach().numpy()
        pred_aux2 = self.aux_seg_2.cpu().detach().numpy()
        pred_aux3 = self.aux_seg_3.cpu().detach().numpy()
        pred_aux4 = self.aux_seg_4.cpu().detach().numpy()
        self.four_predict_map = (pred_aux3+pred_aux4+pred_aux2+pred_aux1)/4.0
        self.four_predict_map[self.four_predict_map > self.pl_threshold] = 1
        self.four_predict_map[self.four_predict_map < 1] = 0
        B,D,W,H = self.four_predict_map.shape
        for i in range(D):
            self.four_predict_map[:,i,:,:] = get_largest_component(self.four_predict_map[:,i,:,:])

    def train_target(self,x):
        self.save_nii(x)
        device = x.device
        pseudo_lab = torch.from_numpy(self.four_predict_map.copy()).float().to(device)  
        size_b,size_c,size_w,size_h = pseudo_lab.shape 

        eara1 = self.aux_seg_1 * pseudo_lab
        eara2 = self.aux_seg_2 * pseudo_lab
        eara3 = self.aux_seg_3 * pseudo_lab
        eara4 = self.aux_seg_4 * pseudo_lab

        diceloss = self.segloss(eara4,pseudo_lab,False)+self.segloss(eara3,pseudo_lab,False) +self.segloss(eara2,pseudo_lab,False)+self.segloss(eara1,pseudo_lab,False)
        diceloss = diceloss / 4.0

        mean_map =  (self.aux_seg_4+self.aux_seg_3 +self.aux_seg_2+self.aux_seg_1) / 4.0
        mean_map_entropyloss = -(mean_map * torch.log2(mean_map + 1e-10)).sum() / (size_c * size_b*size_w*size_h)

        return diceloss, mean_map_entropyloss

def adapt_epoch(upl_model, optim, train_loader, args):
    total_loss = 0.0
    num_batches = 0
    for sample in train_loader:
        imgs = sample['image']
        if torch.cuda.is_available():
            imgs = imgs.cuda()

        # mms volume (B, 3, D, H, W) -> slices (B*D, 3, H, W)
        B, C, D, H, W = imgs.size()
        imgs = imgs.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        # model predict
        diceloss,mean_map_entropyloss = upl_model.train_target(imgs)
        # only basic pseudo-label loss
        loss = diceloss + mean_map_entropyloss
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        total_loss += loss.item()
        num_batches += 1


    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    here = osp.dirname(osp.abspath(__file__))
    args.out = osp.join(here, 'logs_target', args.dataset, args.target, args.note)
    if not osp.exists(args.out):
        os.makedirs(args.out)
    args.out_file = open(osp.join(args.out, 'result.txt'), 'w')
    args.loss_file = open(osp.join(args.out, 'loss.txt'), 'w')
    args.out_file.write(' '.join(sys.argv) + '\n')
    args.out_file.flush()

    # dataset
    img_dir = osp.join(args.data_dir, 'train', 'img', args.target)
    lab_dir = osp.join(args.data_dir, 'train', 'lab', args.target)

    dataset_train = mms_dataloader.niiDataset_2transform_split(source_img=img_dir, source_lab=lab_dir,
                                                         dataset=args.dataset, phase='train', mode='target')
    dataset_val = mms_dataloader.niiDataset_2transform_split(source_img=img_dir, source_lab=lab_dir,
                                                         dataset=args.dataset, phase='val', mode='target')
    dataset_test = mms_dataloader.niiDataset_2transform_split(source_img=img_dir, source_lab=lab_dir,
                                                             dataset=args.dataset, phase='test', mode='target')
    
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2)
    print(f'训练loader的迭代次数 : {len(train_loader)}')
    print(f'验证loader的迭代次数 : {len(val_loader)}')
    print(f'测试loader的迭代次数 : {len(test_loader)}')
    # model  4 classes
    model = netd.DeepLab(num_classes=NUM_CLASSES, backbone='mobilenet', output_stride=args.out_stride,
                           sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)

    log_str = '==> Loading %s model file: %s' % (model.__class__.__name__, args.model_file)
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    checkpoint = torch.load(args.model_file)
    model_state_dict = checkpoint['model_state_dict']
    model_state_dict = {k.replace('module.', '') if k.startswith('module.') else k: v
                        for k, v in model_state_dict.items()}
    filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in model.state_dict()}

    model.load_state_dict(filtered_state_dict, strict=False)

    upl_model = UPLDeepLabV3Plus(model)

    if torch.cuda.is_available():
            upl_model = upl_model.cuda()

    if (args.gpu).find(',') != -1:
        upl_model = torch.nn.DataParallel(upl_model, device_ids=[0, 1])


    optim = torch.optim.Adam(upl_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_decrease_epoch, gamma=args.lr_decrease_rate)

    upl_model.train()

    args.out_file.write(log_str + '\n')
    args.out_file.flush()

    for epoch in range(args.epoch):
        epoch_loss = adapt_epoch(upl_model, optim, train_loader, args)
      
        scheduler.step()

        log_str = 'epoch {}/{}: loss {:.6f}'.format(epoch + 1, args.epoch, epoch_loss)
        print(log_str)
        args.loss_file.write(log_str + '\n')
        args.loss_file.flush()

        if (epoch + 1) % args.interval_valid == 0:
            log_str = '\nepoch {}/{}:'.format(epoch + 1, args.epoch)
            print(log_str)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()

            res_str_t = format_result('UPL Model Val', epoch + 1, *evaluate(upl_model, val_loader))
            print(res_str_t)
            args.out_file.write(res_str_t + '\n')
            args.out_file.flush()

            # ----------------------------------- 
            res_str_t = format_result('UPL Model Test', epoch + 1, *evaluate(upl_model, test_loader))
            print(res_str_t)
            args.out_file.write(res_str_t + '\n')
            args.out_file.flush()
            

def format_result(name, epoch, dice_means, dice_stds, assd_means, assd_stds,
                  hd95_means, hd95_stds, mean_dice, mean_dice_std,
                  mean_assd, mean_assd_std, mean_hd95, mean_hd95_std):
    res_str = '{} {} Dice - LV: {:.2f}\u00b1{:.2f} MYO: {:.2f}\u00b1{:.2f} RV: {:.2f}\u00b1{:.2f}, Mean (LV,MYO,RV): {:.2f}\u00b1{:.2f}'.format(
        name, epoch,
        dice_means[0] * 100, dice_stds[0] * 100, dice_means[1] * 100, dice_stds[1] * 100,
        dice_means[2] * 100, dice_stds[2] * 100, mean_dice * 100, mean_dice_std * 100)
    res_str += '\n{} {} ASSD - LV: {:.4f}\u00b1{:.4f} MYO: {:.4f}\u00b1{:.4f} RV: {:.4f}\u00b1{:.4f}, Mean (LV,MYO,RV): {:.4f}\u00b1{:.4f}'.format(
        name, epoch,
        assd_means[0], assd_stds[0], assd_means[1], assd_stds[1], assd_means[2], assd_stds[2],
        mean_assd, mean_assd_std)
    res_str += '\n{} {} HD95 - LV: {:.4f}\u00b1{:.4f} MYO: {:.4f}\u00b1{:.4f} RV: {:.4f}\u00b1{:.4f}, Mean (LV,MYO,RV): {:.4f}\u00b1{:.4f}'.format(
        name, epoch,
        hd95_means[0], hd95_stds[0], hd95_means[1], hd95_stds[1], hd95_means[2], hd95_stds[2],
        mean_hd95, mean_hd95_std)
    return res_str

if __name__ == '__main__':
    main()