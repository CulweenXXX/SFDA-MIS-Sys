import torch
import numpy as np
import surface_distance as surfdist
import torch.nn.functional as F

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


def _dice_volume(mask_pred, mask_gt):
    """体数据 3D Dice；两者皆空时返回 NaN（对齐 CCRC cal_dice_score 语义）。"""
    mask_pred = np.asarray(mask_pred, dtype=bool)
    mask_gt = np.asarray(mask_gt, dtype=bool)
    denom = mask_pred.sum() + mask_gt.sum()
    if denom == 0:
        return np.nan
    inter = np.logical_and(mask_pred, mask_gt).sum()
    return 2.0 * inter / denom


def _assd_volume(mask_pred, mask_gt):
    """体数据 3D ASSD（spacing=(1,1,1)）；任一方为空时返回 NaN。"""
    mask_pred = np.asarray(mask_pred, dtype=bool)
    mask_gt = np.asarray(mask_gt, dtype=bool)
    if mask_pred.sum() == 0 or mask_gt.sum() == 0:
        return np.nan
    sd = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0))
    val = surfdist.compute_average_surface_distance(sd)
    if isinstance(val, (tuple, list)):
        val = 0.5 * (float(val[0]) + float(val[1]))
    return float(val)


def _hd95_volume(mask_pred, mask_gt):
    """体数据 3D HD95（spacing=(1,1,1)）；任一方为空时返回 NaN。"""
    mask_pred = np.asarray(mask_pred, dtype=bool)
    mask_gt = np.asarray(mask_gt, dtype=bool)
    if mask_pred.sum() == 0 or mask_gt.sum() == 0:
        return np.nan
    sd = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0))
    val = surfdist.compute_robust_hausdorff(sd, 95)
    if isinstance(val, (tuple, list)):
        val = 0.5 * (float(val[0]) + float(val[1]))
    return float(val)


def evaluate(model, val_loader,num_classes,names):
    """按 patient-volume 口径评测（对齐 CCRC `collect` + utils/metric2.py）。

    要求评测 loader 的 batch_size=1（每个 batch 是一个 volume 的全部切片，
    即各方法脚本里 SliceLoaderWrapper 的输出），或直接传 5D (B,3,D,H,W)。
    流程：切片拼回体数据 -> 丢弃 GT 全空切片 -> 逐 volume 计算体 Dice/ASSD/HD95
    -> 对 volume 取 nanmean/nanstd。
    """
    model.eval()
    NUM_CLASSES = num_classes
    NAMES = names

    val_dice = {name: [] for name in NAMES[1:]}
    val_assd = {name: [] for name in NAMES[1:]}
    val_hd95 = {name: [] for name in NAMES[1:]}

    with torch.no_grad():
        for sample in val_loader:
            image = sample['image'].cuda()
            label = sample['label'].cuda()

            if image.dim() == 5:
                B, C, D, H, W = image.shape
                image = image.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
                label = label.permute(0, 2, 1, 3, 4).contiguous().view(B * D, 1, H, W)
            else:
                B = 1
                D = image.shape[0]

            logit, feat = model(image)
            pred = logit.argmax(dim=1).cpu()
            gt = label.long().cpu()
            if gt.dim() == 4:
                gt = gt.squeeze(1)

            for b in range(B):
                vp = pred[b * D:(b + 1) * D]
                vg = gt[b * D:(b + 1) * D]

                keep = (vg != 0).reshape(D, -1).sum(dim=1) > 0
                if int(keep.sum()) == 0:
                    continue

                vp = vp[keep].numpy()
                vg = vg[keep].numpy()

                for c in range(1, NUM_CLASSES):
                    name = NAMES[c]
                    p_mask = vp == c
                    g_mask = vg == c
                    val_dice[name].append(_dice_volume(p_mask, g_mask))
                    val_assd[name].append(_assd_volume(p_mask, g_mask))
                    val_hd95[name].append(_hd95_volume(p_mask, g_mask))

    dice_means = np.array([np.nanmean(val_dice[name]) for name in val_dice], dtype=np.float64)
    dice_stds = np.array([np.nanstd(val_dice[name]) for name in val_dice], dtype=np.float64)
    assd_means = np.array([np.nanmean(val_assd[name]) for name in val_assd], dtype=np.float64)
    assd_stds = np.array([np.nanstd(val_assd[name]) for name in val_assd], dtype=np.float64)
    hd95_means = np.array([np.nanmean(val_hd95[name]) for name in val_hd95], dtype=np.float64)
    hd95_stds = np.array([np.nanstd(val_hd95[name]) for name in val_hd95], dtype=np.float64)

    mean_dice = float(np.nanmean(dice_means))
    mean_dice_std = float(np.nanstd(dice_means))
    mean_assd = float(np.nanmean(assd_means))
    mean_assd_std = float(np.nanstd(assd_means))
    mean_hd95 = float(np.nanmean(hd95_means))
    mean_hd95_std = float(np.nanstd(hd95_means))

    model.train()
    return (dice_means, dice_stds, assd_means, assd_stds, hd95_means, hd95_stds,
            mean_dice, mean_dice_std, mean_assd, mean_assd_std, mean_hd95, mean_hd95_std)

def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`. 

    Args:
        mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
        mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
        the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    mask_gt = np.asarray(mask_gt, dtype=bool)
    mask_pred = np.asarray(mask_pred, dtype=bool)
    volume_gt = mask_gt.sum(axis=(1,2))
    volume_pred = mask_pred.sum(axis=(1,2))
    volume_intersect = (mask_gt & mask_pred).sum(axis=(1,2))

    denom = volume_gt + volume_pred
    dice = np.where(denom > 0, 2 * volume_intersect / denom, np.float64(np.nan))

    return dice

def compute_assd_coefficient(mask_gt, mask_pred, spacing_mm=(1.0, 1.0)):
    """
    mask_gt, mask_pred : [B, H, W] ndarray / tensor-like
    返回 : [B] float，某样本不可算时为 nan
    """
    mask_gt = np.asarray(mask_gt,dtype=bool)
    mask_pred = np.asarray(mask_pred,dtype=bool)

    B = mask_gt.shape[0]
    out = np.full(B, np.nan, dtype=np.float32)

    for i in range(B):
        g = mask_gt[i]
        p = mask_pred[i]

        if g.sum() == 0 or p.sum() == 0:
            continue

        '''
            sd:表面距离的中间计算结果，里面保存了 GT 表面到 Pred 表面、Pred 表面到 GT 表面的逐点最近距离
            val：从 sd 里算出来的两个方向的平均表面距离，是一个二元组 (gt→pred 平均距离, pred→gt 平均距离)
        '''
        sd = surfdist.compute_surface_distances(g, p, spacing_mm=spacing_mm)
        val = surfdist.compute_average_surface_distance(sd)

        # 兼容返回 tuple 的版本
        if isinstance(val, (tuple, list)):
            val = 0.5 * (float(val[0]) + float(val[1]))

        out[i] = float(val)

    return out

def compute_hd95_coefficient(mask_gt, mask_pred, spacing_mm=(1.0, 1.0)):
    """
    mask_gt, mask_pred : [B, H, W] ndarray / tensor-like
    返回 : [B] float，某样本不可算时为 nan
    """
    mask_gt = np.asarray(mask_gt,dtype=bool)
    mask_pred = np.asarray(mask_pred,dtype=bool)

    B = mask_gt.shape[0]
    out = np.full(B, np.nan, dtype=np.float32)

    for i in range(B):
        g = mask_gt[i]
        p = mask_pred[i]

        if g.sum() == 0 or p.sum() == 0:
            continue

        
        sd = surfdist.compute_surface_distances(g, p, spacing_mm=spacing_mm)
        val = surfdist.compute_robust_hausdorff(sd, 95)

        # 兼容返回 tuple 的版本
        if isinstance(val, (tuple, list)):
            val = 0.5 * (float(val[0]) + float(val[1]))

        out[i] = float(val)

    return out


def _to_onehot(pred, target, num_classes):
    """pred:[B,C,H,W] logits, target:[B,1,H,W] -> two [B,C,H,W] numpy bool"""
    pred_idx   = pred.argmax(dim=1)                                # (B,H,W)
    pred_oh    = F.one_hot(pred_idx, num_classes).permute(0,3,1,2) # (B,C,H,W)
    target_idx = target.squeeze(1).long()
    target_oh  = F.one_hot(target_idx, num_classes).permute(0,3,1,2)
    return pred_oh.cpu().numpy().astype(bool), target_oh.cpu().numpy().astype(bool)


def _per_channel(fn, pred, target, num_classes, names, skip_bg=True):
    """对每个通道跑一次 fn，返回 {name: [B] array}"""
    pred_oh, target_oh = _to_onehot(pred, target, num_classes)
    start = 1 if skip_bg else 0
    return {
        names[c]: fn(pred_oh[:, c], target_oh[:, c])
        for c in range(start, num_classes)
    }


# ---------- 三个公开接口，签名一致 ----------
def dice_onehot(pred, target, num_classes, names):
    '''
        pred : prediction of model, the shape is [B, C, H, W]
        target : groundtruth of samples, the shape is [B, 1, H, W]
        num_classes : the number of segmentation classes (include background)
        names : ['BG', 'FG1', ... ]
    '''
    return _per_channel(compute_dice_coefficient, pred, target, num_classes, names)

def assd_onehot(pred, target, num_classes, names):
    return _per_channel(compute_assd_coefficient, pred, target, num_classes, names)

def hd95_onehot(pred, target, num_classes, names):
    return _per_channel(compute_hd95_coefficient, pred, target, num_classes, names)

