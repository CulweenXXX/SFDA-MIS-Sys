from scipy import ndimage
import numpy as np
def evaluate(model, val_loader):
    model.eval()

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

def get_largest_component(image):
    """
    get the largest component from 2D or 3D binary image
    image: nd array
    """
    dim = len(image.shape)
    if(image.sum() == 0 ):
        # print('the largest component is null')
        return image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return  output


def tensor_rot_90(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3).transpose(2, 3)
    else:
	    return x.flip(2).transpose(1, 2)
def tensor_rot_180(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3).flip(2)
    else:
	    return x.flip(2).flip(1)
def tensor_flip_2(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(2)
    else:
	    return x.flip(1)
def tensor_flip_3(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3)
    else:
	    return x.flip(2)

def tensor_rot_270(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.transpose(2, 3).flip(3)
    else:
        return x.transpose(1, 2).flip(2)
    
def rotate_single_with_label(img, label):
    if label == 1:
        img = tensor_rot_90(img)
    elif label == 2:
        img = tensor_rot_180(img)
    elif label == 3:
        img = tensor_rot_270(img)
    else:
        img = img
    return img