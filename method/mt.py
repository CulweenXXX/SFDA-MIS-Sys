import argparse
from torch import Tensor
from utils.metrics import *

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
torch.backends.cudnn.enabled = False
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloaders import mms_dataloader
from datetime import datetime
import networks.deeplabv3 as netd
import torch.backends.cudnn as cudnn
import random
import sys

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

def soft_label_to_hard(soft_pls, pseudo_label_threshold):
    pseudo_labels = torch.zeros(soft_pls.size())
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[soft_pls > pseudo_label_threshold] = 1
    pseudo_labels[soft_pls <= pseudo_label_threshold] = 0

    return pseudo_labels

@torch.no_grad()
def update_ema(teacher, student, decay):
    # 更新参数
    for t_p, s_p in zip(teacher.parameters(), student.parameters()):
        t_p.data = decay * t_p.data + (1 - decay) * s_p.data
    

def adapt_epoch(model_t, model_s, optim, train_loader, args):
    total_loss = 0.0
    num_batches = 0
    for sample_w, sample_s in train_loader:
        imgs_w = sample_w['image']
        imgs_s = sample_s['image']
        if torch.cuda.is_available():
            imgs_w = imgs_w.cuda()
            imgs_s = imgs_s.cuda()

        # mms volume (B, 3, D, H, W) -> slices (B*D, 3, H, W)
        B, C, D, H, W = imgs_w.size()
        imgs_w = imgs_w.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        imgs_s = imgs_s.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)

        # model predict
        predictions_stu_s = model_s(imgs_s)['out']
        with torch.no_grad():
            predictions_tea_w = model_t(imgs_w)['out']

        ###softmax
        # predictions_stu_s_softmax: Tensor = F.softmax(predictions_stu_s, dim=1)
        predictions_tes_w_softmax: Tensor = F.softmax(predictions_tea_w, dim=1)
        pseudo_label = torch.argmax(predictions_tes_w_softmax,dim=1)

        # only basic pseudo-label loss
        loss = F.cross_entropy(predictions_stu_s, pseudo_label)
        # loss = F.cross_entropy(predictions_stu_s, pseudo_label,reduction='none')
        # loss = (loss * mask).sum() / mask.sum()

        loss.backward()
        optim.step()
        optim.zero_grad()
        total_loss += loss.item()
        num_batches += 1
        # update teacher
        update_ema(model_t,model_s,decay=args.model_ema_rate)
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    now = datetime.now()
    here = osp.dirname(osp.abspath(__file__))
    args.out = osp.join(here, 'logs_target', args.dataset, args.target, args.note)
    if not osp.exists(args.out):
        os.makedirs(args.out)
    args.out_file = open(osp.join(args.out, 'result.txt'), 'w')
    args.loss_file = open(osp.join(args.out, 'loss.txt'), 'w')
    args.out_file.write(' '.join(sys.argv) + '\n')
    # args.out_file.write(print_args(args) + '\n')
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
    model_s = netd.DeepLab(num_classes=NUM_CLASSES, backbone='mobilenet', output_stride=args.out_stride,
                           sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    model_t = netd.DeepLab(num_classes=NUM_CLASSES, backbone='mobilenet', output_stride=args.out_stride,
                           sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)

    if torch.cuda.is_available():
        model_s = model_s.cuda()
        model_t = model_t.cuda()
    log_str = '==> Loading %s model file: %s' % (model_s.__class__.__name__, args.model_file)
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    checkpoint = torch.load(args.model_file)
    model_state_dict = checkpoint['model_state_dict']
    model_state_dict = {k.replace('module.', '') if k.startswith('module.') else k: v
                        for k, v in model_state_dict.items()}
    filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in model_s.state_dict()}
    model_s.load_state_dict(filtered_state_dict, strict=False)
    model_t.load_state_dict(filtered_state_dict, strict=False)

    if (args.gpu).find(',') != -1:
        model_s = torch.nn.DataParallel(model_s, device_ids=[0, 1])
        model_t = torch.nn.DataParallel(model_t, device_ids=[0, 1])

    optim = torch.optim.Adam(model_s.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_decrease_epoch, gamma=args.lr_decrease_rate)

    model_s.train()
    model_t.train()
    for param in model_t.parameters():
        param.requires_grad = False

    args.out_file.write(log_str + '\n')
    args.out_file.flush()

    for epoch in range(args.epoch):
        epoch_loss = adapt_epoch(model_t, model_s, optim, train_loader, args)
      
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

            res_str_t = format_result('Teacher Val', epoch + 1, *evaluate(model_t, val_loader))
            print(res_str_t)
            args.out_file.write(res_str_t + '\n')
            args.out_file.flush()

            res_str_s = format_result('Student Val', epoch + 1, *evaluate(model_s, val_loader))
            print(res_str_s)
            args.out_file.write(res_str_s + '\n')
            args.out_file.flush()

            # ----------------------------------- 
            res_str_t = format_result('Teacher Test', epoch + 1, *evaluate(model_t, test_loader))
            print(res_str_t)
            args.out_file.write(res_str_t + '\n')
            args.out_file.flush()
            
            res_str_s = format_result('Student Test', epoch + 1, *evaluate(model_s, test_loader))
            print(res_str_s)
            args.out_file.write(res_str_s + '\n')
            args.out_file.flush()


if __name__ == '__main__':
    main()