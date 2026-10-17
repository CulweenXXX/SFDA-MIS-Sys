import argparse
from torch import Tensor
from utils.metrics import *
from utils.metrics import evaluate as _metrics_evaluate
from utils.tools import get_largest_component

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('--model-file', type=str, default='logs_train/mms/checkpoint_80.pth.tar')  # source weight path

parser.add_argument('--model', type=str, default='Deeplab', help='Deeplab')
parser.add_argument('--out-stride', type=int, default=16)
parser.add_argument('--sync-bn', type=bool, default=True)
parser.add_argument('--freeze-bn', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr-decrease-rate', type=float, default=0.9, help='ratio multiplied to initial lr')
parser.add_argument('--lr-decrease-epoch', type=int, default=4, help='interval epoch number for lr decrease')

parser.add_argument('--data-dir', default='/data0/grchen/Data/M&MS')
parser.add_argument('--dataset', type=str, default='mms')
parser.add_argument('--target', type=str, default='B', help='target domain (B/C/D)')
parser.add_argument('--split', type=str, default='train', choices=['train', 'valid', 'test'],
                    help='which split to evaluate on (target domains only have labeled data under train/)')
parser.add_argument('--batch-size', type=int, default=2)

parser.add_argument('--sam-code-root', type=str, default='/data0/grchen/IPLC',
                    help='root dir containing the segment_anything package of SAM-Med2D')
parser.add_argument('--sam-model-file', type=str, default='/data0/grchen/IPLC/pretrain_model/sam-med2d_b.pth',
                    help='SAM-Med2D checkpoint path')
parser.add_argument('--sample-times', type=int, default=10, help='SAM sampling times of each slice')
parser.add_argument('--curve-loss-weight', type=float, default=0.01)
parser.add_argument('--note', type=str, default='descript')
parser.add_argument('--interval-valid',type=int,default=1)
args = parser.parse_args()

import os

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import os.path as osp

import numpy as np
import cv2
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


sam_predictor = None


def get_sam_predictor(args):
    global sam_predictor
    if sam_predictor is None:
        if args.sam_code_root not in sys.path:
            sys.path.insert(0, args.sam_code_root)
        from external.segment_anything import sam_model_registry
        from external.segment_anything.predictor_sammed import SammedPredictor
        sam_args = argparse.Namespace()
        sam_args.image_size = 256
        sam_args.encoder_adapter = True
        sam_args.sam_checkpoint = args.sam_model_file
        sam_model = sam_model_registry['vit_b'](sam_args)
        if torch.cuda.is_available():
            sam_model = sam_model.cuda()
        sam_model.eval()
        sam_predictor = SammedPredictor(sam_model)
    return sam_predictor


def norm_0_255(x):
    normalized_x = ((x + 1) / 2) * 255
    normalized_x = normalized_x.to(torch.uint8)
    return normalized_x.cpu().detach().numpy()


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


def random_sample(LV):
    LV[LV == 1] = 255
    LV = LV.astype(np.uint8)
    white_indices_LV = np.where(LV == 255)
    points = []
    input_labels = []
    process_part_1(white_indices_LV, points, input_labels)
    if all(points_part is None for points_part in points):
        return None
    all_points = np.concatenate([points_part for points_part in points if points_part is not None])
    all_input_labels = np.concatenate([input_labels_part for input_labels_part in input_labels if input_labels_part is not None])
    return all_points, all_input_labels
    

def generate_sam_pl(probs, images, predictor, num_classes, sample_times):
    """IPLC pseudo-label generation: refine the model argmax mask with SAM-Med2D.

    probs: (N, C, H, W) softmax probabilities of the student model
    images: (N, 3, H, W) input slices in [-1, 1]
    """
    mask_input = probs.copy()
    output = np.argmax(probs, axis=1)
    dec_output = np.zeros((probs.shape[0], num_classes, 256, 256), dtype=np.uint8)
    for c in range(num_classes):
        dec_output[:, c, :, :] = (output.copy() == c)
    argmax_output = dec_output.copy()

    images_all_depth = norm_0_255(images)
    all_depth_weight_entropy = np.zeros((probs.shape[0], 256, 256), dtype=np.float64)
    sam_pseudo_label = np.zeros((probs.shape[0], num_classes, 256, 256), dtype=np.uint8)

    for depth in range(probs.shape[0]):
        for c in range(num_classes):
            argmax_output[depth, c, :, :] = get_largest_component(argmax_output[depth, c, :, :])
        images_one_depth = np.zeros((256, 256, 3), dtype=np.uint8)
        images_one_depth[:, :, 0] = images_all_depth[depth, 0, :, :]
        images_one_depth[:, :, 1] = images_all_depth[depth, 0, :, :]
        images_one_depth[:, :, 2] = images_all_depth[depth, 0, :, :]
        predictor.set_image(images_one_depth.copy())

        mask_input_this_depth = np.zeros((num_classes, 64, 64), dtype=np.float64)
        for c in range(num_classes):
            resized_mask = cv2.resize(mask_input[depth, c, :, :], (64, 64), interpolation=cv2.INTER_LINEAR)
            mask_input_this_depth[c, :, :] = resized_mask
        mask_input_this_depth = mask_input_this_depth[np.newaxis, :, :, :]

        this_depth_mask_all = np.zeros((sample_times, num_classes, 256, 256), dtype=np.float64)
        this_depth_probability_map = np.zeros((num_classes, 256, 256), dtype=np.float64)

        for i in range(sample_times):
            for c in range(num_classes - 1):
                result = random_sample(argmax_output[depth, c + 1, :, :].copy())
                if result is not None:
                    points, labels = result
                    masks, scores, logits = predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        mask_input=mask_input_this_depth[:, c + 1, :, :],
                        multimask_output=True,
                    )
                else:
                    masks, scores, logits = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        mask_input=mask_input_this_depth[:, c + 1, :, :],
                        multimask_output=True,
                    )
                this_depth_mask_all[i, c + 1, :, :] = logits.copy()

            Fore = np.sum(this_depth_mask_all[i, :, :, :], axis=0)
            Back = 1 - Fore
            Back[Back < 0] = 0
            this_depth_mask_all[i, 0, :, :] = Back

        for c in range(num_classes):
            this_depth_probability_map[c, :, :] = np.sum(this_depth_mask_all[:, c, :, :], axis=0) / sample_times

        this_depth_entropy = -np.sum(this_depth_probability_map.copy() * np.log2(this_depth_probability_map.copy() + 1e-10), axis=0)
        this_depth_weight_entropy = (2 - this_depth_entropy) / 2
        all_depth_weight_entropy[depth, :, :] = this_depth_weight_entropy.copy()

        this_depth_argmax_map = np.argmax(this_depth_probability_map.copy(), axis=0)
        for c in range(num_classes):
            sam_pseudo_label[depth, c, :, :] = (this_depth_argmax_map == c)
        for c in range(num_classes):
            sam_pseudo_label[depth, c, :, :] = get_largest_component(sam_pseudo_label[depth, c, :, :])

    return sam_pseudo_label, all_depth_weight_entropy


def dice_loss(predict, target, ent_dice_weight_map):
    target = target.float()
    smooth = 1e-4
    intersect = torch.sum(ent_dice_weight_map * predict * target * ent_dice_weight_map)
    dice = (2 * intersect + smooth) / (torch.sum(target * ent_dice_weight_map) + torch.sum(predict * ent_dice_weight_map) + smooth)
    loss = 1.0 - dice
    return loss


def weighted_dice_loss(predict, target, ent_dice_weight_map):
    loss = 0.0
    for i in range(NUM_CLASSES):
        loss += dice_loss(predict[:, i, :, :], target[:, i, :, :], ent_dice_weight_map)
    return loss / NUM_CLASSES


def curvature_loss(output):
    """
    Args:
        output: (B, C, H, W) model predictions (probabilities after softmax)
    Returns:
        curvature loss (scalar)
    """
    device = output.device
    B, C, H, W = output.shape

    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    kernel_x = torch.tensor([[-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 1, 3)
    kernel_y = torch.tensor([[-1], [0], [1]], dtype=torch.float32, device=device).view(1, 1, 3, 1)
    kernel_xx = torch.tensor([[-1, 2, -1]], dtype=torch.float32, device=device).view(1, 1, 1, 3)
    kernel_yy = torch.tensor([[-1], [2], [-1]], dtype=torch.float32, device=device).view(1, 1, 3, 1)

    total_loss = 0.0

    for b in range(B):
        for c in range(1, C):
            prob = output[b, c, :, :]
            prob = prob.unsqueeze(0).unsqueeze(0)

            edge_x = F.conv2d(prob, sobel_x, padding=1)
            edge_y = F.conv2d(prob, sobel_y, padding=1)
            edge_map = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)

            U_x = F.conv2d(edge_map, kernel_x, padding=(0, 1))
            U_y = F.conv2d(edge_map, kernel_y, padding=(1, 0))

            U_xx = F.conv2d(U_x, kernel_xx, padding=(0, 1))
            U_yy = F.conv2d(U_y, kernel_yy, padding=(1, 0))
            U_xy = F.conv2d(U_x, kernel_y, padding=(1, 0))

            numerator = (U_xx * (1 + U_y**2) - 2 * U_xy * U_x * U_y + U_yy * (1 + U_x**2))
            denominator = 2 * (1 + U_x**2 + U_y**2) ** (3 / 2)
            curvature = numerator / (denominator + 1e-8)

            negative_curv = F.relu(-curvature)
            mean_neg_curv = torch.sum(negative_curv) / (H * W)

            total_loss += mean_neg_curv

    if B == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / B


def adapt_epoch(model, optim, train_loader, args):
    predictor = get_sam_predictor(args)
    total_loss = 0.0
    num_batches = 0
    for _, sample in train_loader:
        imgs = sample['image']
        if torch.cuda.is_available():
            imgs = imgs.cuda()

        # mms volume (B, 3, D, H, W) -> slices (B*D, 3, H, W)
        B, C, D, H, W = imgs.size()
        imgs = imgs.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)

        # model predict (single forward, softmax is reused by both losses)
        logits = model(imgs)['out']
        probs = F.softmax(logits, dim=1)

        # IPLC: SAM-Med2D refined pseudo label + entropy weight map
        sam_pl, weight_map = generate_sam_pl(probs.detach().cpu().numpy(), imgs,
                                             predictor, NUM_CLASSES, args.sample_times)
        sam_pl = torch.from_numpy(sam_pl).float().to(probs.device)
        weight_map = torch.from_numpy(weight_map).float().to(probs.device)

        seg_loss = weighted_dice_loss(probs, sam_pl, weight_map)
        curve_loss = curvature_loss(probs)
        loss = seg_loss + args.curve_loss_weight * curve_loss

        loss.backward()
        optim.step()
        optim.zero_grad()
        total_loss += loss.item()
        num_batches += 1
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

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_decrease_epoch, gamma=args.lr_decrease_rate)

    model.train()

    args.out_file.write(log_str + '\n')
    args.out_file.flush()

    for epoch in range(args.epoch):
        epoch_loss = adapt_epoch(model, optim, train_loader, args)
      
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

            res_str = format_result('Val', epoch + 1, *evaluate(model, val_loader))
            print(res_str)
            args.out_file.write(res_str + '\n')
            args.out_file.flush()

            # ----------------------------------- 
            res_str = format_result('Test', epoch + 1, *evaluate(model, test_loader))
            print(res_str)
            args.out_file.write(res_str + '\n')
            args.out_file.flush()


if __name__ == '__main__':
    main()