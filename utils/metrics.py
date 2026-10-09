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


def evaluate(model, val_loader,num_classes,names):
    model.eval()
    NUM_CLASSES = num_classes
    NAMES = names

    val_dice = {name: [] for name in NAMES[1:]}
    val_assd = {name: [] for name in NAMES[1:]}
    val_hd95 = {name: [] for name in NAMES[1:]}

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            image = sample[0].cuda()
            label = sample[1].cuda()

            logit,feat = model(image)
            prob = F.softmax(logit,dim=1)               # [B,5,H,W] ---> [B,5,H,W]


            dice = dice_onehot(prob, label, NUM_CLASSES, names=NAMES)
            assd = assd_onehot(prob, label, NUM_CLASSES, names=NAMES)
            hd95 = hd95_onehot(prob, label, NUM_CLASSES, names=NAMES)

            for name in val_dice:
                val_dice[name].extend(np.asarray(dice[name], dtype=np.float64).tolist())
                val_assd[name].extend(np.asarray(assd[name], dtype=np.float64).tolist())
                val_hd95[name].extend(np.asarray(hd95[name], dtype=np.float64).tolist())

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

