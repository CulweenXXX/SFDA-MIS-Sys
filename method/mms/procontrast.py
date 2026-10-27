import argparse
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

parser.add_argument('--pfa-epoch', type=int, default=5, help='epochs of prototype-anchored feature alignment stage')
parser.add_argument('--pfa-lr', type=float, default=1e-4)
parser.add_argument('--pfa-lr-decrease-rate', type=float, default=0.8, help='ratio multiplied to initial lr')
parser.add_argument('--pfa-lr-decrease-epoch', type=int, default=1, help='interval epoch number for lr decrease')
parser.add_argument('--cl-epoch', type=int, default=5, help='epochs of contrastive learning stage')
parser.add_argument('--cl-lr', type=float, default=1e-4)
parser.add_argument('--cl-lr-decrease-rate', type=float, default=0.95, help='ratio multiplied to initial lr')
parser.add_argument('--cl-lr-decrease-epoch', type=int, default=2, help='interval epoch number for lr decrease')
parser.add_argument('--weight-decay', type=float, default=5e-4)

parser.add_argument('--data-dir', default='/data0/grchen/Data/M&MS')
parser.add_argument('--dataset', type=str, default='mms')
parser.add_argument('--target', type=str, default='B', help='target domain (B/C/D)')
parser.add_argument('--split', type=str, default='train', choices=['train', 'valid', 'test'],
                    help='which split to evaluate on (target domains only have labeled data under train/)')
parser.add_argument('--batch-size', type=int, default=2)

parser.add_argument('--nav-t', type=float, default=0.1, help='temperature of the prototype transport loss')
parser.add_argument('--proto-beta', type=float, default=0.001, help='momentum for updating target proportions')
parser.add_argument('--s-par', type=float, default=0.5, help='weight of the prototype-to-target direction')

parser.add_argument('--use-pseudo', action='store_true', help='enable the pseudo-label loss in the CL stage')
parser.add_argument('--drop-percent', type=float, default=80)
parser.add_argument('--use-source-prototypes', action='store_true')
parser.add_argument('--momentum-prototype', action='store_true')
parser.add_argument('--num-queries', type=int, default=64)
parser.add_argument('--num-negatives', type=int, default=256)
parser.add_argument('--current-class-threshold', type=float, default=0.3)
parser.add_argument('--low-entropy-threshold', type=float, default=20)
parser.add_argument('--low-rank', type=int, default=2)
parser.add_argument('--high-rank', type=int, default=5)
parser.add_argument('--queue-size', type=int, default=30000)
parser.add_argument('--bg-queue-size', type=int, default=50000)

parser.add_argument('--note', type=str, default='descript')
parser.add_argument('--interval-valid', type=int, default=1)
parser.add_argument('--skip-empty-slices', type=str2bool, default=True,
                    help='skip slices without foreground GT during evaluation')

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
from torch.utils.data import DataLoader
from dataloaders import mms_dataloader
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
    return _metrics_evaluate(EvalWrapper(model), SliceLoaderWrapper(val_loader), NUM_CLASSES, NAMES,
                             skip_empty_slices=args.skip_empty_slices)


class ProtoLoss(nn.Module):
    """PCT (NeurIPS 2021) prototype transport loss used by the PFA stage."""

    def __init__(self, nav_t, beta, num_classes, device, s_par=0.5):
        super().__init__()
        self.nav_t = nav_t
        self.s_par = s_par
        self.beta = beta
        self.prop = (torch.ones((num_classes, 1)) * (1 / num_classes)).to(device)
        self.eps = 1e-6

    def pairwise_cosine_dist(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return 1 - torch.matmul(x, y.T)

    def get_pos_logits(self, sim_mat, prop):
        log_prior = torch.log(prop + self.eps)
        return sim_mat / self.nav_t + log_prior

    def update_prop(self, prop):
        return (1 - self.beta) * self.prop + self.beta * prop

    def forward(self, mu_s, f_t):
        sim_mat = torch.matmul(mu_s, f_t.T)

        old_logits = self.get_pos_logits(sim_mat.detach(), self.prop)
        s_dist_old = F.softmax(old_logits, dim=0)
        prop = s_dist_old.mean(1, keepdim=True)
        self.prop = self.update_prop(prop)

        new_logits = self.get_pos_logits(sim_mat, self.prop)
        s_dist = F.softmax(new_logits, dim=0)
        t_dist = F.softmax(sim_mat / self.nav_t, dim=1)
        cost_mat = self.pairwise_cosine_dist(mu_s, f_t)

        t2p_loss = (self.s_par * cost_mat * s_dist).sum(0).mean()
        p2t_loss = (((1 - self.s_par) * cost_mat * t_dist).sum(1) * self.prop.squeeze(1)).sum()
        return t2p_loss, p2t_loss


class PseudoLabelLoss(nn.Module):
    """Entropy-filtered pseudo-label CE loss used by the CL stage."""

    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, pred, target, drop_percent, prob_teacher):
        b, c, h, w = pred.shape
        with torch.no_grad():
            entropy = -torch.sum(prob_teacher * torch.log(prob_teacher + self.eps), dim=1)
            for i in range(c):
                if torch.sum(entropy[target == i]) > 10:
                    thresh = np.percentile(
                        entropy[target == i].detach().cpu().numpy().flatten(), drop_percent)
                    thresh_mask = entropy.ge(thresh).bool() * (target == i).bool()
                    target[thresh_mask] = 255
        weight = b * h * w / torch.sum(target != 255)
        return weight * F.cross_entropy(pred, target, ignore_index=255)


class PrototypeContrastiveLoss(nn.Module):
    """Memory-bank prototype contrastive loss used by the CL stage."""

    def __init__(self, feature_dim, args, device):
        super().__init__()
        self.args = args
        self.memobank = []
        self.queue_sizes = []
        for i in range(NUM_CLASSES):
            self.memobank.append([torch.zeros(0, feature_dim)])
            self.queue_sizes.append(args.bg_queue_size if i == 0 else args.queue_size)
        if args.momentum_prototype:
            self.momentum_prototype = torch.zeros(
                (NUM_CLASSES, args.num_queries, 1, feature_dim), device=device)
        else:
            self.momentum_prototype = None

    @torch.no_grad()
    def enqueue(self, keys, class_idx):
        if keys.shape[0] == 0:
            return
        queue = torch.cat((self.memobank[class_idx][0], keys.detach().cpu()), dim=0)
        if queue.shape[0] > self.queue_sizes[class_idx]:
            queue = queue[-self.queue_sizes[class_idx]:]
        self.memobank[class_idx][0] = queue

    def forward(self, target_f, target_f_teacher, pseudo_onehot, prob_teacher,
                low_mask, high_mask, source_prototypes):
        args = self.args
        num_feat = target_f.shape[1]
        low_valid_pixel = pseudo_onehot * low_mask

        target_f = target_f.permute(0, 2, 3, 1)
        target_f_teacher = target_f_teacher.permute(0, 2, 3, 1)

        _, prob_indices_teacher = torch.sort(prob_teacher, 1, True)
        prob_indices_teacher = prob_indices_teacher.permute(0, 2, 3, 1)

        valid_classes = []
        seg_num_list = []
        seg_feat_low_entropy = [None] * NUM_CLASSES
        seg_proto = [None] * NUM_CLASSES

        for i in range(NUM_CLASSES):
            low_valid_pixel_seg = low_valid_pixel[:, i]
            prob_seg = prob_teacher[:, i]
            low_entropy_seg = (prob_seg > args.current_class_threshold) * low_valid_pixel_seg.bool()

            seg_feat_low_entropy[i] = target_f[low_entropy_seg]

            class_mask = torch.sum(
                prob_indices_teacher[:, :, :, args.low_rank:args.high_rank].eq(i), dim=3).bool()
            negative_mask = high_mask[:, 0].bool() * class_mask
            self.enqueue(target_f_teacher[negative_mask].detach(), i)

            num_valid = int(low_valid_pixel_seg.sum().item())
            if num_valid > 0:
                seg_num_list.append(num_valid)
                valid_classes.append(i)
                seg_proto[i] = target_f_teacher[low_valid_pixel_seg.bool()].detach().mean(0, keepdim=True)

        if len(seg_num_list) <= 1:
            return torch.tensor(0.0, device=target_f.device) * target_f.sum()

        reco_loss = torch.tensor(0.0, device=target_f.device)
        valid_seg = len(seg_num_list)

        for k in range(valid_seg):
            cls = valid_classes[k]
            anchors = seg_feat_low_entropy[cls]
            if anchors is None or anchors.shape[0] == 0 or self.memobank[cls][0].shape[0] == 0:
                reco_loss = reco_loss + 0 * target_f.sum()
                continue

            anchor_idx = torch.randint(anchors.shape[0], size=(args.num_queries,), device=target_f.device)
            anchor_feat = anchors[anchor_idx]

            with torch.no_grad():
                negative_feat = self.memobank[cls][0].to(target_f.device)
                neg_idx = torch.randint(negative_feat.shape[0],
                                        size=(args.num_queries * args.num_negatives,),
                                        device=target_f.device)
                negative_feat = negative_feat[neg_idx].view(args.num_queries, args.num_negatives, num_feat)

                if args.use_source_prototypes:
                    positive_feat = source_prototypes[cls].view(1, 1, -1).repeat(args.num_queries, 1, 1)
                else:
                    positive_feat = seg_proto[cls].unsqueeze(0).repeat(args.num_queries, 1, 1)
                    if self.momentum_prototype is not None:
                        if self.momentum_prototype[cls].abs().sum() > 0:
                            positive_feat = 0.001 * positive_feat + 0.999 * self.momentum_prototype[cls]
                        self.momentum_prototype[cls] = positive_feat.clone()

            all_feat = torch.cat((positive_feat, negative_feat), dim=1)
            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
            target = torch.zeros(args.num_queries, dtype=torch.long, device=target_f.device)
            reco_loss = reco_loss + F.cross_entropy(seg_logits / 0.1, target)

        return reco_loss / valid_seg


def compute_entropy_masks(prob_teacher, pseudo_label, step, total_steps, args):
    entropy = -torch.sum(prob_teacher * torch.log(prob_teacher + 1e-10), dim=1)
    alpha_t = args.low_entropy_threshold * (1 - step / total_steps)

    low_mask = torch.zeros_like(pseudo_label, dtype=torch.bool)
    high_mask = torch.zeros_like(pseudo_label, dtype=torch.bool)

    for i in range(NUM_CLASSES):
        class_entropy = entropy[pseudo_label == i]
        if class_entropy.numel() == 0 or class_entropy.sum() <= 5:
            continue
        if i == 0:
            low_thresh, high_thresh = np.percentile(
                class_entropy.detach().cpu().numpy().flatten(), [90 - alpha_t / 4, 100 - alpha_t / 4])
        else:
            low_thresh, high_thresh = np.percentile(
                class_entropy.detach().cpu().numpy().flatten(), [90 - alpha_t, 100 - alpha_t])
        low_mask |= entropy.le(low_thresh) & (pseudo_label == i)
        high_mask |= entropy.ge(high_thresh) & (pseudo_label == i)

    return low_mask.unsqueeze(1), high_mask.unsqueeze(1)


def pfa_adapt_epoch(model, prototypes, criterion, optim, train_loader, args):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for sample_w, sample_s in train_loader:
        optim.zero_grad()

        imgs = sample_w['image']
        if torch.cuda.is_available():
            imgs = imgs.cuda()

        # mms volume (B, 3, D, H, W) -> slices (B*D, 3, H, W)
        B, C, D, H, W = imgs.size()
        imgs = imgs.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)

        feat = model(imgs)['feat']
        features = feat.permute(0, 2, 3, 1).contiguous().view(-1, feat.size(1))

        t2p_loss, p2t_loss = criterion(prototypes, features)
        loss = t2p_loss + p2t_loss

        loss.backward()
        optim.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0


def cl_adapt_epoch(model_s, model_t, contra_loss_fn, criterion_pseudo, optim,
                   train_loader, step, total_steps, source_prototypes, args):
    model_s.train()
    model_t.eval()
    total_loss = 0.0
    num_batches = 0
    for sample_w, sample_s in train_loader:
        optim.zero_grad()

        imgs = sample_w['image']
        if torch.cuda.is_available():
            imgs = imgs.cuda()

        # mms volume (B, 3, D, H, W) -> slices (B*D, 3, H, W)
        B, C, D, H, W = imgs.size()
        imgs = imgs.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)

        with torch.no_grad():
            out_t = model_t(imgs)
            feat_t = out_t['feat']
            prob_t = F.softmax(out_t['out'], dim=1)
            pseudo_t = torch.argmax(prob_t, dim=1)

        out_s = model_s(imgs)
        feat_s = out_s['feat']
        logits_s = out_s['out']

        loss = torch.tensor(0.0, device=feat_s.device)
        if args.use_pseudo:
            percent_unreliable = (100 - args.drop_percent) * (1 - step / total_steps)
            drop_percent = 100 - percent_unreliable
            loss = loss + criterion_pseudo(logits_s, pseudo_t.clone(), drop_percent, prob_t.detach())

        prob_t_feat = F.interpolate(prob_t, size=feat_t.shape[2:], mode='bilinear', align_corners=True)
        pseudo_feat = torch.argmax(prob_t_feat, dim=1)
        low_mask, high_mask = compute_entropy_masks(prob_t_feat, pseudo_feat, step, total_steps, args)
        pseudo_onehot = F.one_hot(pseudo_feat, NUM_CLASSES).permute(0, 3, 1, 2)
        loss = loss + contra_loss_fn(feat_s, feat_t, pseudo_onehot, prob_t_feat,
                                     low_mask, high_mask, source_prototypes)

        loss.backward()
        optim.step()
        total_loss += loss.item()
        num_batches += 1
        step += 1
    return (total_loss / num_batches if num_batches > 0 else 0.0), step


def build_model(args):
    model = netd.DeepLab(num_classes=NUM_CLASSES, backbone='mobilenet', output_stride=args.out_stride,
                         sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    if torch.cuda.is_available():
        model = model.cuda()
    checkpoint = torch.load(args.model_file)
    model_state_dict = checkpoint['model_state_dict']
    model_state_dict = {k.replace('module.', '') if k.startswith('module.') else k: v
                        for k, v in model_state_dict.items()}
    filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict, strict=False)
    if (args.gpu).find(',') != -1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    return model


def base_model_of(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_dict(model, state):
    if state is not None:
        model.load_state_dict(state)


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

    here = '/data0/grchen/SFDA_MIS_TestSys'
    args.out = osp.join(here, 'logs_target', args.dataset, args.target, args.note, fold_dir)
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
    model = build_model(args)
    base_model = base_model_of(model)
    log_str = '==> Loading %s model file: %s' % (model.__class__.__name__, args.model_file)
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()

    # the source classifier is frozen and its weights serve as prototypes
    for param in base_model.decoder.last_conv_2.parameters():
        param.requires_grad = False
    source_prototypes = base_model.decoder.last_conv_2[1].weight.detach().view(NUM_CLASSES, -1).clone()

    res_str = format_result('Source Only', 0, *evaluate(model, test_loader))
    print(res_str)
    args.out_file.write(res_str + '\n')
    args.out_file.flush()

    # ------------------------- PFA stage -------------------------
    criterion_proto = ProtoLoss(args.nav_t, args.proto_beta, NUM_CLASSES,
                                torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                args.s_par)
    pfa_params = [p for p in model.parameters() if p.requires_grad]
    pfa_optim = torch.optim.Adam(pfa_params, lr=args.pfa_lr, betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)
    pfa_scheduler = torch.optim.lr_scheduler.StepLR(pfa_optim, step_size=args.pfa_lr_decrease_epoch,
                                                    gamma=args.pfa_lr_decrease_rate)

    best_pfa_dice = -1.0
    best_pfa_epoch = 0
    best_pfa_state = None
    best_pfa_metrics = None

    for epoch in range(args.pfa_epoch):
        epoch_loss = pfa_adapt_epoch(model, source_prototypes, criterion_proto, pfa_optim, train_loader, args)
        pfa_scheduler.step()

        log_str = 'PFA epoch {}/{}: loss {:.6f}'.format(epoch + 1, args.pfa_epoch, epoch_loss)
        print(log_str)
        args.loss_file.write(log_str + '\n')
        args.loss_file.flush()

        val_metrics = evaluate(model, val_loader)
        val_dice = float(val_metrics[6])
        if val_dice > best_pfa_dice:
            best_pfa_dice = val_dice
            best_pfa_epoch = epoch + 1
            best_pfa_state = clone_state_dict(model)
            best_pfa_metrics = val_metrics

        log_str = '\nPFA epoch {}/{}:'.format(epoch + 1, args.pfa_epoch)
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        res_str = format_result('PFA Val', epoch + 1, *val_metrics)
        print(res_str)
        args.out_file.write(res_str + '\n')
        args.out_file.flush()

        if (epoch + 1) % args.interval_valid == 0:
            args.out_file.write('-'*32 + '\n')
            args.out_file.flush()

            res_str = format_result('PFA Test', epoch + 1, *evaluate(model, test_loader))
            print(res_str)
            args.out_file.write(res_str + '\n')
            args.out_file.flush()

    if best_pfa_state is not None:
        load_state_dict(model, best_pfa_state)
        log_str = '\nPFA best checkpoint: epoch {}, val Dice {:.2f}'.format(
            best_pfa_epoch, best_pfa_dice * 100)
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        res_str = format_result('PFA Best Val', best_pfa_epoch, *best_pfa_metrics)
        print(res_str)
        args.out_file.write(res_str + '\n')
        args.out_file.flush()

        res_str = format_result('PFA Best Test', best_pfa_epoch, *evaluate(model, test_loader))
        print(res_str)
        args.out_file.write(res_str + '\n')
        args.out_file.flush()

    # ------------------------- CL stage -------------------------
    teacher = netd.DeepLab(num_classes=NUM_CLASSES, backbone='mobilenet', output_stride=args.out_stride,
                           sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    teacher_state_dict = {k.replace('module.', ''): v for k, v in model.state_dict().items()}
    teacher.load_state_dict(teacher_state_dict)
    if torch.cuda.is_available():
        teacher = teacher.cuda()
    if (args.gpu).find(',') != -1:
        teacher = torch.nn.DataParallel(teacher, device_ids=[0, 1])
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    criterion_pseudo = PseudoLabelLoss()
    contra_loss_fn = PrototypeContrastiveLoss(source_prototypes.shape[1], args,
                                              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    cl_params = [p for p in model.parameters() if p.requires_grad]
    cl_optim = torch.optim.Adam(cl_params, lr=args.cl_lr, betas=(0.9, 0.999),
                                weight_decay=args.weight_decay)
    cl_scheduler = torch.optim.lr_scheduler.StepLR(cl_optim, step_size=args.cl_lr_decrease_epoch,
                                                   gamma=args.cl_lr_decrease_rate)

    total_steps = max(args.cl_epoch * len(train_loader), 1)
    step = 0
    patience = 2
    bad_epochs = 0
    best_cl_dice = -1.0
    best_cl_epoch = 0
    best_cl_state = None
    best_cl_metrics = None

    for epoch in range(args.cl_epoch):
        epoch_loss, step = cl_adapt_epoch(model, teacher, contra_loss_fn, criterion_pseudo, cl_optim,
                                          train_loader, step, total_steps, source_prototypes, args)
        cl_scheduler.step()

        log_str = 'CL epoch {}/{}: loss {:.6f}'.format(epoch + 1, args.cl_epoch, epoch_loss)
        print(log_str)
        args.loss_file.write(log_str + '\n')
        args.loss_file.flush()

        val_metrics = evaluate(model, val_loader)
        val_dice = float(val_metrics[6])
        if val_dice > best_cl_dice:
            best_cl_dice = val_dice
            best_cl_epoch = epoch + 1
            best_cl_state = clone_state_dict(model)
            best_cl_metrics = val_metrics
            bad_epochs = 0
        else:
            bad_epochs += 1

        log_str = '\nCL epoch {}/{}:'.format(epoch + 1, args.cl_epoch)
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        res_str = format_result('CL Val', epoch + 1, *val_metrics)
        print(res_str)
        args.out_file.write(res_str + '\n')
        args.out_file.flush()

        if (epoch + 1) % args.interval_valid == 0:
            args.out_file.write('-'*32 + '\n')
            args.out_file.flush()

            res_str = format_result('Test', epoch + 1, *evaluate(model, test_loader))
            print(res_str)
            args.out_file.write(res_str + '\n')
            args.out_file.flush()

        if bad_epochs >= patience:
            log_str = 'CL early stop at epoch {}: no val improvement for {} epochs'.format(
                epoch + 1, bad_epochs)
            print(log_str)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            break

    if best_cl_state is not None:
        load_state_dict(model, best_cl_state)
        log_str = '\nCL best checkpoint: epoch {}, val Dice {:.2f}'.format(
            best_cl_epoch, best_cl_dice * 100)
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        res_str = format_result('CL Best Val', best_cl_epoch, *best_cl_metrics)
        print(res_str)
        args.out_file.write(res_str + '\n')
        args.out_file.flush()

        res_str = format_result('CL Best Test', best_cl_epoch, *evaluate(model, test_loader))
        print(res_str)
        args.out_file.write(res_str + '\n')
        args.out_file.flush()


if __name__ == '__main__':
    main()
