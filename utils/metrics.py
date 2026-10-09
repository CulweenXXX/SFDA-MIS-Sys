import torch
import numpy as np
import surface_distance as surfdist
import torch.nn.functional as F

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

