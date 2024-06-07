import SimpleITK as sitk
import numpy as np
import torch, random

import igl
from skimage import measure
from torch import nn
import math
from scipy import ndimage as ndi
import numbers
from torch.nn import functional as F

# feature_sampling: 这里使用最简单的特征提取，双线性插值
class BasicSample(nn.Module):

    def __init__(self):
        super(BasicSample, self).__init__()

    #输入vertices.shape = B,N,3
    # voxel_features = B,C,D,H,W
    def forward(self, voxel_features, vertices):
        neighbourhood = vertices[:, :, None, None]
        features = F.grid_sample(voxel_features, neighbourhood, mode='bilinear', padding_mode='zeros',align_corners=True)
        features = features[:, :, :, 0, 0].transpose(2, 1)
        return features

# target one-hot编码
def to_one_hot_3d(tensor, n_classes=3):  # shape = [batch, s, h, w]
    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot

def random_crop_3d(img, label, crop_size):
    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]
    random_z_max = img.shape[2] - crop_size[2]

    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)
    z_random = random.randint(0, random_z_max)

    crop_img = img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
    crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],z_random:z_random + crop_size[2]]

    return crop_img, crop_label

def center_crop_3d(img, label, slice_num=16):
    if img.shape[0] < slice_num:
        return None
    left_x = img.shape[0]//2 - slice_num//2
    right_x = img.shape[0]//2 + slice_num//2

    crop_img = img[left_x:right_x]
    crop_label = label[left_x:right_x]
    return crop_img, crop_label

def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    return file_name_list

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_V2(optimizer, lr):
    """Sets the learning rate to a fixed number"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_obj(path,seg):
    seg = np.float_(seg)
    seg = ndi.gaussian_filter(seg, sigma=1.0)
    verts, faces, normals, values = measure.marching_cubes(seg, level=0.3)
    # verts, faces, normals, values = measure.marching_cubes(seg)
    igl.write_obj(path, verts, faces)

def save_img(array,path):
    shape = array.shape
    if len(shape)>3:
        for i in range(len(shape)-1):
            array =np.squeeze(array)
    img = sitk.GetImageFromArray(array)
    sitk.WriteImage(img, path)

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

def heatmap(shape,center_index, kernel_size, sigma):
    batch,_ = center_index.shape
    map = torch.zeros(shape)[None].repeat(batch,1,1,1).float().cuda()
    for i in range(batch):
        x,y,z = center_index[i]
        map[i,x,y,z] = 1

    pad = torch.nn.ConstantPad3d(padding=kernel_size//2, value=0).cuda()
    guass = GaussianSmoothing(1,kernel_size,sigma,dim=3).cuda()

    map = pad(map[:,None])
    map = guass.forward(map)
    maxValue = guass.weight.max()
    map /=maxValue
    return map


def Oriented_Bounding_Box(verts):
    # verts --(n,3)
    center = np.mean(verts, axis=0)
    new_verts = verts - center
    U, Sigma, VT = np.linalg.svd(new_verts, full_matrices=False, compute_uv=True)
    verts_pca_svd = np.dot(new_verts, VT.T)

    right_bound = np.max(verts_pca_svd, axis=0)
    right_mat = np.diag(right_bound)
    right_point = np.dot(right_mat, VT)

    left_bound = np.min(verts_pca_svd, axis=0)
    left_mat = np.diag(left_bound)
    left_point = np.dot(left_mat, VT)

    center_bound = (right_bound + left_bound) / 2
    center_point = np.dot(center_bound, VT)
    size_bound = (right_bound - left_bound)

    return center_point + center, VT.T, size_bound

#核大小必须为基数,粗糙的膨胀函数
def tensor_dilate(tensor, size = 3 ,time = 1):
    pool = nn.MaxPool3d(size, stride=1)
    pad = nn.ConstantPad3d(padding = int((size-1)/2),value= 0)
    for i in range(time):
        tensor = pad(tensor)
        tensor = pool(tensor)
    return tensor

def tensor_erose(tensor, size = 3 ,time = 1):
    tensor = -1 * tensor
    tensor =tensor_dilate(tensor, size, time)
    tensor = -1 * tensor
    return tensor

def tensor_close(tensor, size = 3 ,time = 1):
    tensor = tensor_dilate(tensor, size, time)
    tensor = tensor_erose(tensor, size, time)
    return tensor

def centerCrop3d(seg_tensor,crop_size,padValue = 0):
    shape = seg_tensor.shape
    axisNum = len(shape)
    padding = []

    begin = axisNum - 3
    croping = []
    for i in range(3):
        left = 0
        right = 0
        if crop_size[i] > shape[i+begin]:
            left = math.floor((crop_size[i] - shape[i+begin]) / 2)
            right = math.ceil((crop_size[i] - shape[i+begin]) / 2)
        padding.append(left)
        padding.append(right)
    padding.reverse()


    pad = torch.nn.ConstantPad3d(padding, value=padValue)
    seg_tensor = pad(seg_tensor)
    shape = seg_tensor.shape
    croping = []
    for i in range(3):
        rise = shape[i+begin] - crop_size[i]
        left = math.floor(rise / 2)
        right = shape[i+begin] - math.ceil(rise/ 2)
        croping.append((left,right))

    croping = torch.tensor(croping)
    croping[croping<0] = 0
    seg_tensor = seg_tensor[:,croping[0][0]:croping[0][1],croping[1][0]:croping[1][1],croping[2][0]:croping[2][1]]
    return seg_tensor

if __name__ == "__main__":

    a = torch.zeros((1,7,7,7))
    centerCrop3d(a, (9,9,9), padValue=0)


