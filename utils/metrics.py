import torch
import numpy as np
import surface_distance as surfdist
import torch.nn.functional as F

def format_result(name, epoch, class_names, dice_means, dice_stds, assd_means, assd_stds,
                  hd95_means, hd95_stds, mean_dice, mean_dice_std,
                  mean_assd, mean_assd_std, mean_hd95, mean_hd95_std):
    class_names = list(class_names)
    n = len(dice_means)
    if len(class_names) == n + 1:
        class_names = class_names[1:]
    elif len(class_names) > n:
        class_names = class_names[len(class_names) - n:]

    def _format_metric(metric_name, means, stds, mean, mean_std, scale, ndigits):
        per_class = ' '.join(
            '{}: {:.{p}f}\u00b1{:.{p}f}'.format(nm, m * scale, s * scale, p=ndigits)
            for nm, m, s in zip(class_names, means, stds)
        )
        return '{} {} {} - {}, Mean ({}): {:.{p}f}\u00b1{:.{p}f}'.format(
            name, epoch, metric_name, per_class, ','.join(class_names),
            mean * scale, mean_std * scale, p=ndigits)

    res_str = _format_metric('Dice', dice_means, dice_stds, mean_dice, mean_dice_std, 100.0, 2)
    res_str += '\n' + _format_metric('ASSD', assd_means, assd_stds, mean_assd, mean_assd_std, 1.0, 2)
    res_str += '\n' + _format_metric('HD95', hd95_means, hd95_stds, mean_hd95, mean_hd95_std, 1.0, 2)
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


def _model_logits(model, image):
    out = model(image)
    if isinstance(out, dict):
        return out['out']
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def collect_predictions_mms(model, val_loader, skip_empty_slices=True):
    """MMS：dict sample，返回逐 volume 的 logits [C,H,W,D] 与 GT [1,H,W,D]。"""
    model.eval()
    pred_results_list = []
    gt_segs_list = []
    with torch.no_grad():
        for sample in val_loader:
            image = sample['image'].cuda()
            label = sample['label'].cuda().long()

            if image.dim() == 5:
                B, C, D, H, W = image.shape
                image = image.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
                label = label.permute(0, 2, 1, 3, 4).contiguous().view(B * D, 1, H, W)

            predictions = _model_logits(model, image)
            preds, targets = [], []
            for i in range(predictions.shape[0]):
                if skip_empty_slices and label[i].sum() == 0:
                    continue
                preds.append(predictions[i].detach().cpu())
                targets.append(label[i].detach().cpu())

            if len(preds) == 0:
                continue
            pred_results_list.append(torch.stack(preds, dim=-1))
            gt_segs_list.append(torch.stack(targets, dim=-1))

    model.train()
    return pred_results_list, gt_segs_list


def collect_predictions_mmwhs(model, val_loader,skip_empty_slices=False):
    """MM-WHS：按该数据集的 sample 格式实现。"""
    model.eval()
    case_slices = {}  # case_name -> [(slice_idx, pred_cls[H,W], gt_cls[H,W]), ...]

    with torch.no_grad():
        for sample in val_loader:
            image = sample['image'].cuda()
            label = sample['label'].cuda().long()

            predictions = _model_logits(model, image)

            pred_cls = predictions.detach().cpu()
            gt_cls = label.detach().cpu()

            for i in range(pred_cls.shape[0]):
                case_slices.setdefault(sample['case_name'][i], []).append(
                    (int(sample['slice_idx'][i]), pred_cls[i], gt_cls[i]))

    pred_results_list = []
    gt_segs_list = []
    for case_name in sorted(case_slices):
        slices = sorted(case_slices[case_name], key=lambda item: item[0])
        pred_vol =torch.stack([item[1] for item in slices], dim=-1)  # [C,H,W,D]
        gt_vol = torch.stack([item[2] for item in slices], dim=-1)    # [H,W,D]

        pred_results_list.append(pred_vol)
        gt_segs_list.append(gt_vol)
        
    return pred_results_list, gt_segs_list

def collect_predictions_pnp(model, data_loader,skip_empty_slices=False):
    model.eval()
    # 用 List 收集所有的 Batch
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for sample in data_loader:
            data = sample['image'].cuda()
            target_map = sample['label']
            
            # 获取网络输出并转类别 [B, C, H, W] -> [B, H, W]
            predictions = _model_logits(model, data)
            pred_cls = predictions.detach().cpu()
            gt_cls = target_map.cpu()

            all_preds.append(pred_cls)
            all_gts.append(gt_cls)
            
    # 在切片维度（维度0）拼接
    # 假设 loader 是严格按顺序读取的，拼接后就是完整的 3D 体积
    pred_vol = torch.cat(all_preds, dim=0)   # [N*600, H, W]
    gt_vol   = torch.cat(all_gts,   dim=0)
    
    pred_cases = list(torch.split(pred_vol, 600, dim=0))   # list of [600, H, W]
    gt_cases   = list(torch.split(gt_vol,   600, dim=0))

    pred_results_list = [p.permute(1, 2, 3, 0).contiguous() for p in pred_cases]   # [C,H,W,N_i]
    gt_segs_list   = [g.permute(1, 2, 0).contiguous()    for g in gt_cases]     # [H,W,N_i]

    return pred_results_list, gt_segs_list

def _to_index_volume(pred, gt):
    if torch.is_tensor(pred):
        pred = pred.detach().cpu()
        if pred.dim() == 4:
            pred = pred.argmax(dim=0)
        pred = pred.numpy()
    else:
        pred = np.asarray(pred)
        if pred.ndim == 4:
            pred = pred.argmax(axis=0)

    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()
    else:
        gt = np.asarray(gt)
    while gt.ndim > 3 and gt.shape[0] == 1:
        gt = gt[0]
    return pred, gt


def evaluate(model, val_loader, num_classes, names, skip_empty_slices=True,
             collect_fn=collect_predictions_mms):
    """统一评估入口，由 collect_fn 负责各数据集的预测收集。

    collect_fn(model, val_loader, skip_empty_slices) -> (pred_list, gt_list)，
    每个元素是一个 volume：pred 为 [C,H,W,D] logits 或 [D,H,W] 索引，
    gt 为 [1,H,W,D] 或 [D,H,W]。
    """
    model.eval()
    NUM_CLASSES = num_classes
    NAMES = names
    val_dice = {name: [] for name in NAMES[1:]}
    val_assd = {name: [] for name in NAMES[1:]}
    val_hd95 = {name: [] for name in NAMES[1:]}

    pred_results_list, gt_segs_list = collect_fn(model, val_loader, skip_empty_slices)
  
    for pred, gt in zip(pred_results_list, gt_segs_list):
        pred, gt = _to_index_volume(pred, gt)       # [H,W,D] and [H,W,D]
        for c in range(1, NUM_CLASSES):
            name = NAMES[c]
            p_mask = pred == c
            g_mask = gt == c
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
    return (NAMES, dice_means, dice_stds, assd_means, assd_stds, hd95_means, hd95_stds,
            mean_dice, mean_dice_std, mean_assd, mean_assd_std, mean_hd95, mean_hd95_std)
