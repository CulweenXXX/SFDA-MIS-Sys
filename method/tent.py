import argparse
from torch import Tensor
from utils.metrics import *
from utils.metrics import evaluate as _metrics_evaluate

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('--model-file', type=str, default='logs_train/mms/checkpoint_80.pth.tar')  # source weight path

parser.add_argument('--model', type=str, default='Deeplab', help='Deeplab')
parser.add_argument('--out-stride', type=int, default=16)
parser.add_argument('--sync-bn', type=bool, default=True)
parser.add_argument('--freeze-bn', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=20)

parser.add_argument('--data-dir', default='/data0/grchen/Data/M&MS')
parser.add_argument('--dataset', type=str, default='mms')
parser.add_argument('--target', type=str, default='B', help='target domain (B/C/D)')
parser.add_argument('--split', type=str, default='train', choices=['train', 'valid', 'test'],
                    help='which split to evaluate on (target domains only have labeled data under train/)')
parser.add_argument('--batch-size', type=int, default=2)

parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--steps', type=int, default=1, help='gradient steps of tent on each batch')
parser.add_argument('--episodic', action='store_true', help='reset model/optimizer before each batch')
parser.add_argument('--note', type=str, default='descript')
parser.add_argument('--interval-valid',type=int,default=1)

parser.add_argument('--fold-json', type=str, default=None,
                    help='K-fold 划分文件路径，如 kfold_splits/mms/kfold_B.json')
parser.add_argument('--fold-id', type=int, default=None,
                    help='当前折号（0 ~ K-1）')
args = parser.parse_args()

import os

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import os.path as osp
import json

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
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torch.backends.cudnn as cudnn
import random
import sys
from copy import deepcopy

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

class EvalWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        output = self.model(image)
        return output['out'], output['feat']


class SliceLoaderWrapper:
    """Flatten (B, 3, D, H, W) volumes to (B*D, 3, H, W) slices for evaluation.

    4D samples are passed through unchanged.
    """
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for sample in self.loader:
            image = sample['image']
            label = sample['label']
            if image.dim() == 5:
                B, C, D, H, W = image.shape
                image = image.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
                label = label.permute(0, 2, 1, 3, 4).contiguous().view(B * D, 1, H, W)
            sample = dict(sample)
            sample['image'] = image
            sample['label'] = label
            yield sample

    def __len__(self):
        return len(self.loader)


def evaluate(model, val_loader):
    return _metrics_evaluate(EvalWrapper(model), SliceLoaderWrapper(val_loader), NUM_CLASSES, NAMES)


# ------------------------- tent core -------------------------
# Reference: https://github.com/DequanWang/tent

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)['out']
    # adapt (mean over all pixels instead of .mean(0), which is scalar only for 1D predictions)
    loss = softmax_entropy(outputs).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, SynchronizedBatchNorm2d)):
            for np_, p in m.named_parameters():
                if np_ in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent.

    Note: tent's original configure_model also disables the running stats
    (track_running_stats=False, running_mean/var=None) to force batch statistics
    in eval mode. That breaks here: SynchronizedBatchNorm2d passes running stats
    through unconditionally (eval forward would raise), and the ASPP global
    average pooling normalizes a (B, C, 1, 1) tensor, which batch-stat mode
    rejects for B == 1. Running stats are therefore kept (and keep adapting).
    """
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, SynchronizedBatchNorm2d)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, (nn.BatchNorm2d, SynchronizedBatchNorm2d)) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"


def adapt_epoch(tent, train_loader, args):
    total_loss = 0.0
    num_batches = 0
    for _, sample in train_loader:
        imgs = sample['image']
        if torch.cuda.is_available():
            imgs = imgs.cuda()

        # mms volume (B, 3, D, H, W) -> slices (B*D, 3, H, W)
        B, C, D, H, W = imgs.size()
        imgs = imgs.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)

        # tent: forward + entropy minimization step(s) on the target batch
        outputs = tent(imgs)
        loss = softmax_entropy(outputs).mean()

        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0


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

        # 输出目录：{note}/fold_{fold_id}
        fold_dir = f'fold_{args.fold_id}'
    else:
        train_patients = None
        val_patients   = None
        test_patients  = None
        fold_dir = 'full'

    now = datetime.now()
    here = '/data0/grchen/SFDA_MIS_TestSys'
    args.out = osp.join(here, 'logs_target', args.dataset, args.target, args.note, fold_dir)
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
                                                         dataset=args.dataset, phase='train', mode='target',
                                                         patient_list=train_patients)
    dataset_val = mms_dataloader.niiDataset_2transform_split(source_img=img_dir, source_lab=lab_dir,
                                                         dataset=args.dataset, phase='val', mode='target',
                                                         patient_list=val_patients)
    dataset_test = mms_dataloader.niiDataset_2transform_split(source_img=img_dir, source_lab=lab_dir,
                                                             dataset=args.dataset, phase='test', mode='target',
                                                             patient_list=test_patients)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2)

    print(f'训练loader的迭代次数 : {len(train_loader)}')
    print(f'验证loader的迭代次数 : {len(val_loader)}')
    print(f'测试loader的迭代次数 : {len(test_loader)}')

    # model  4 classes
    model = netd.DeepLab(num_classes=NUM_CLASSES, backbone='mobilenet', output_stride=args.out_stride,
                         sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)

    if torch.cuda.is_available():
        model = model.cuda()
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

    if (args.gpu).find(',') != -1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    # tent: only the affine params of the norm layers are adapted
    configure_model(model)
    check_model(model)
    params, names = collect_params(model)
    if args.optimizer == 'SGD':
        optim = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
    else:
        optim = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    tent = Tent(model, optim, steps=args.steps, episodic=args.episodic)

    log_str = 'tent: adapt %d / %d params' % (sum(p.numel() for p in params),
                                              sum(p.numel() for p in model.parameters()))
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()

    res_str = format_result('Source Only', 0, *evaluate(model, test_loader))
    print(res_str)
    args.out_file.write(res_str + '\n')
    args.out_file.flush()

    for epoch in range(args.epoch):
        epoch_loss = adapt_epoch(tent, train_loader, args)

        log_str = 'epoch {}/{}: loss {:.6f}'.format(epoch + 1, args.epoch, epoch_loss)
        print(log_str)
        args.loss_file.write(log_str + '\n')
        args.loss_file.flush()

        if (epoch + 1) % args.interval_valid == 0:
            log_str = '\nepoch {}/{}:'.format(epoch + 1, args.epoch)
            print(log_str)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()

            res_str = format_result('Val', epoch + 1, *evaluate(model, val_loader))
            print(res_str)
            args.out_file.write(res_str + '\n')
            args.out_file.flush()

            args.out_file.write('-'*32 + '\n')
            args.out_file.flush()

            res_str = format_result('Test', epoch + 1, *evaluate(model, test_loader))
            print(res_str)
            args.out_file.write(res_str + '\n')
            args.out_file.flush()


if __name__ == '__main__':
    main()
