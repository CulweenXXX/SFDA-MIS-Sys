from scipy import ndimage
import numpy as np

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