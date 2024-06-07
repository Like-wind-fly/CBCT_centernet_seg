import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')
# Preprocess parameters
parser.add_argument('--classNum', type=int, default=1,help='number of classes')  #分割类别数（只分割牙与非牙，类别为2）
# 将灰度值归一化到[0,1]
parser.add_argument('--upper', type=int, default=2500, help='')
parser.add_argument('--lower', type=int, default=-500, help='')
parser.add_argument('--median', type=int, default=1000, help='')
parser.add_argument('--norm_factor', type=float, default=1500, help='')

# 这是图像大小的归一化
parser.add_argument('--expand_slice', type=int, default=16, help='')
parser.add_argument('--min_slices', type=int, default=32, help='')
parser.add_argument('--xy_down_scale', type=float, default=1, help='')
parser.add_argument('--slice_down_scale', type=float, default=1, help='')
parser.add_argument('--valid_rate', type=float, default=0.2, help='')
parser.add_argument('--min_size', type=list, default=[128,256,256])




# 关于heatmap 图的提取
parser.add_argument('--down_ratio', type=int, default=4, help='')   #下采样率
# 使用高斯卷积的大小
parser.add_argument('--kernel_size', type=int, default=5, help='')
# 标准差 sigma
parser.add_argument('--sigma', type=float, default=0.8, help='')


# data in/out and dataset
parser.add_argument('--dataset_path',default = '/home/zhangzechu/workspace/data/CBCT',help='data path ,for segmentation and volume')
parser.add_argument('--save',default='ResUNet',help='save path of trained model')

# train
parser.add_argument('--train_crop_size', type=list, default=[128,128,128])
parser.add_argument('--val_crop_max_size', type=int, default=128)
# 图片增强的参数
parser.add_argument('--scale_factor', type=float, default=0.1, help='')


# testing
parser.add_argument('--test_crop_size', type=list, default=[160,256,256])
args = parser.parse_args()


