import argparse
from torch import Tensor
from utils.metrics import *
from utils.metrics import evaluate as _metrics_evaluate
from utils.tools import str2bool

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

parser.add_argument('--fold-json', type=str, default=None,
                    help='K-fold 划分文件路径，如 ./kfold_splits/mms/kfold_D.json')
parser.add_argument('--fold-id', type=int, default=None,
                    help='当前折号（0 ~ K-1）')

parser.add_argument('--cons-lambda',type=float,default=1.0)
parser.add_argument('--skip-empty-slices', type=str2bool, default=True,
                    help='skip slices without foreground GT during evaluation')
parser.add_argument('--pl-source', type=str, default='teacher', choices=['teacher', 'mean'],
                    help='pseudo-label source: teacher or mean of teacher+students')
args = parser.parse_args()

import os

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import os.path as osp

import numpy as np
import torch.nn.functional as F

import torch
torch.backends.cudnn.enabled = False
from torch.autograd import Variable
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataloaders import mms_dataloader
from datetime import datetime
import networks.deeplabv3 as netd
import torch.backends.cudnn as cudnn
import random
import sys
import itertools
import matplotlib.pyplot as plt

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
NAMES = ['BG', 'LV', 'MYO', 'RV']


def soft_label_to_hard(soft_pls, pseudo_label_threshold):
    pseudo_labels = torch.zeros(soft_pls.size())
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[soft_pls > pseudo_label_threshold] = 1
    pseudo_labels[soft_pls <= pseudo_label_threshold] = 0

    return pseudo_labels

@torch.no_grad()
def update_ema(teacher, student1, student2, alpha, beta):
    # 更新参数
    for t_p, s1_p, s2_p in zip(teacher.parameters(), student1.parameters(), student2.parameters()):
        t_p.data = alpha * t_p.data + (1 - alpha) * (beta * s1_p.data + (1.0 - beta) * s2_p.data)

def supervised_loss(logits, pseudo_label):
    loss_map = F.cross_entropy(logits, pseudo_label, reduction='none')
    return loss_map.mean()


def adapt_epoch(epoch, model_t, model_s1, model_s2, optim, train_loader, args):
    total_loss = 0.0
    ce_loss = 0.0
    cons_loss = 0.0
    num_batches = 0

    for sample_w, sample_s in train_loader:
        imgs_w = sample_w['image']
        imgs_s = sample_s['image']


        if torch.cuda.is_available():
            imgs_w = imgs_w.cuda()
            imgs_s = imgs_s.cuda()

        # (B, C, D, H, W) -> (B*D, C, H, W)
        B, C, D, H, W = imgs_w.size()

        imgs_w = imgs_w.permute(0, 2, 1, 3, 4).contiguous()
        imgs_w = imgs_w.view(B * D, C, H, W)

        imgs_s = imgs_s.permute(0, 2, 1, 3, 4).contiguous()
        imgs_s = imgs_s.view(B * D, C, H, W)

        optim.zero_grad(set_to_none=True)

        # --------------------------------------------------
        # 1. Teacher: generate semantic pseudo labels
        # --------------------------------------------------
        with torch.no_grad():
            logits_t = model_t(imgs_w)['out']
            prob_t = F.softmax(logits_t, dim=1)

        # --------------------------------------------------
        # 2. Student 1: semantic learning
        # --------------------------------------------------
        logits_s1 = model_s1(imgs_w)['out']

        # --------------------------------------------------
        # 3. Student 2: semantic learning + consistency
        # --------------------------------------------------
        logits_s2_w = model_s2(imgs_w)['out']
        logits_s2_s = model_s2(imgs_s)['out']

        if args.pl_source == 'mean':
            prob_pl = (prob_t + F.softmax(logits_s1, dim=1)
                       + F.softmax(logits_s2_w, dim=1)) / 3.0
        else:
            prob_pl = prob_t
        pseudo_label = prob_pl.argmax(dim=1)

        loss_ce_s1 = supervised_loss(
            logits_s1, pseudo_label
        )

        # Semantic supervision on the weak view
        loss_ce_s2 = supervised_loss(
            logits_s2_w, pseudo_label
        )

        # Weak-view prediction serves as a detached reference
        with torch.no_grad():
            prob_s2_w = F.softmax(logits_s2_w, dim=1)
    
        # KL(P_weak || P_strong)
        log_prob_s2_s = F.log_softmax(
            logits_s2_s, dim=1
        )

        loss_cons = F.kl_div(
            log_prob_s2_s,
            prob_s2_w,
            reduction='none'
        ).sum(dim=1).mean()

        # --------------------------------------------------
        # 4. Total loss
        # --------------------------------------------------
        loss_ce = 0.5 * loss_ce_s1 + 0.5 * loss_ce_s2

        loss = loss_ce + args.cons_lambda * loss_cons

        loss.backward()
        optim.step()

        ce_loss += loss_ce.item()
        cons_loss += loss_cons.item() * args.cons_lambda
        total_loss += loss.item()
        num_batches += 1
        # update teacher
        update_ema(model_t,model_s1,model_s2,alpha=args.model_ema_rate,beta=0.5)

    return (ce_loss / num_batches if num_batches > 0 else 0.0,
            cons_loss / num_batches if num_batches > 0 else 0.0,
            total_loss / num_batches if num_batches > 0 else 0.0)

def evaluate(model, data_loader):
    return _metrics_evaluate(model, data_loader, NUM_CLASSES, NAMES,
                             skip_empty_slices=args.skip_empty_slices)



import json
def main():
    # ---------- 读取 K-fold 划分 ----------
    if args.fold_json is not None and args.fold_id is not None:
        with open(args.fold_json, 'r') as f:
            folds = json.load(f)
        fold = folds[args.fold_id]
        train_patients = fold['train']
        val_patients   = fold['val']
        test_patients  = fold['test']

        print(f'=== Fold {args.fold_id} ===')
        print(f'train patients ({len(train_patients)}): {train_patients}')
        print(f'val patients   ({len(val_patients)}): {val_patients}')
        print(f'test patients  ({len(test_patients)}): {test_patients}')

        # 安全检查
        assert len(set(train_patients) & set(val_patients)) == 0
        assert len(set(train_patients) & set(test_patients)) == 0
        assert len(set(val_patients)   & set(test_patients)) == 0
        print('no patient overlap')

        fold_dir = f'fold_{args.fold_id}'
    else:
        train_patients = None
        val_patients   = None
        test_patients  = None
        fold_dir = 'full'

    here = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
    args.out = osp.join(here, 'logs_target', args.dataset, args.target,
                    args.note, fold_dir)
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


    dataset_train = mms_dataloader.niiDataset_2transform_split(
        source_img=img_dir, source_lab=lab_dir,
        dataset=args.dataset, phase='train', mode='target',
        patient_list=train_patients)          # ← 加这行
    dataset_val = mms_dataloader.niiDataset_2transform_split(
        source_img=img_dir, source_lab=lab_dir,
        dataset=args.dataset, phase='val', mode='target',
        patient_list=val_patients)            # ← 加这行
    dataset_test = mms_dataloader.niiDataset_2transform_split(
        source_img=img_dir, source_lab=lab_dir,
        dataset=args.dataset, phase='test', mode='target',
        patient_list=test_patients)           # ← 加这行
    
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2)

    # model  4 classes
    model_s1 = netd.DeepLab(num_classes=NUM_CLASSES, backbone='mobilenet', output_stride=args.out_stride,
                               sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    model_s2 = netd.DeepLab(num_classes=NUM_CLASSES, backbone='mobilenet', output_stride=args.out_stride,
                                sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    model_t = netd.DeepLab(num_classes=NUM_CLASSES, backbone='mobilenet', output_stride=args.out_stride,
                            sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)

    if torch.cuda.is_available():
        model_s1 = model_s1.cuda()
        model_s2 = model_s2.cuda()
        model_t = model_t.cuda()
    log_str = '==> Loading %s model file: %s' % (model_t.__class__.__name__, args.model_file)
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    checkpoint = torch.load(args.model_file)
    model_state_dict = checkpoint['model_state_dict']
    model_state_dict = {k.replace('module.', '') if k.startswith('module.') else k: v
                        for k, v in model_state_dict.items()}
    filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in model_t.state_dict()}

    model_s1.load_state_dict(filtered_state_dict, strict=False)
    model_s2.load_state_dict(filtered_state_dict, strict=False)
    model_t.load_state_dict(filtered_state_dict, strict=False)

    if (args.gpu).find(',') != -1:
        model_s1 = torch.nn.DataParallel(model_s1, device_ids=[0, 1])
        model_s2 = torch.nn.DataParallel(model_s2, device_ids=[0, 1])
        model_t = torch.nn.DataParallel(model_t, device_ids=[0, 1])

    optim = torch.optim.Adam(itertools.chain(model_s1.parameters(),model_s2.parameters()), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_decrease_epoch, gamma=args.lr_decrease_rate)

    model_s1.train()
    model_s2.train()
    model_t.train()
    for param in model_t.parameters():
        param.requires_grad = False

    res_str = format_result('Source Only', 0, *evaluate(model_t, test_loader))
    print(res_str)
    args.out_file.write(res_str + '\n')
    args.out_file.flush()


    for epoch in range(args.epoch):
        ce_loss, cons_loss, epoch_loss = adapt_epoch(epoch, model_t, model_s1, model_s2, optim, train_loader, args)
        
        scheduler.step()

        log_str = 'epoch {}/{}: ce_loss {:.6f} cons_loss {:.6f} total_loss {:.6f}'.format(
            epoch + 1, args.epoch, ce_loss, cons_loss, epoch_loss)
        print(log_str)
        args.loss_file.write(log_str + '\n')
        args.loss_file.flush()

        if (epoch + 1) % args.interval_valid == 0:
            log_str = '\nepoch {}/{}:'.format(epoch + 1, args.epoch)
            print(log_str)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()

            res_str_t = format_result('Teacher val', epoch + 1, *evaluate(model_t, val_loader))
            print(res_str_t)
            args.out_file.write(res_str_t + '\n')
            args.out_file.flush()

            res_str_t = format_result('Teacher Test', epoch + 1, *evaluate(model_t, test_loader))
            print(res_str_t)
            args.out_file.write(res_str_t + '\n')
            args.out_file.flush()


if __name__ == '__main__':
    main()