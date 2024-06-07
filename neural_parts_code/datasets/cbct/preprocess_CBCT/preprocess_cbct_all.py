# 这个预处理只是将nii格式的文件，转为容易读取写的.pickle文件
# 会将图像的长宽高统一步长，删除掉多余的部分，缩放为统一的大小，对于小于标准的大小的图像进行
# 填充。
#具体参数定义在config.py
import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join
import scipy.stats as st
import config
import math
import torch
import pickle

class DataModes:
    TRAINING = 'training'
    VALIDATION = 'validation'
    TESTING = 'testing'
    ALL = 'all'
    def __init__(self):
        dataset_splits = [DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]

def splitDataset(dataset_path):
    file_list = os.listdir(dataset_path)
    file_list = [os.path.join(dataset_path,fname) for fname in file_list if "pickle" in fname]
    toothNum = len(file_list)

    print('Total numbers of samples is :', toothNum)
    np.random.seed(0)
    perm = np.random.permutation(toothNum)
    counts = [perm[:int(toothNum * 0.8)], perm[int(toothNum * 0.8):int(toothNum * 0.9)],perm[int(toothNum * 0.9):int(toothNum)]]

    dataset = {}

    dataset[DataModes.TRAINING] = [file_list[i] for i in counts[0]]
    dataset[DataModes.VALIDATION] = [file_list[i] for i in counts[1]]
    dataset[DataModes.TESTING] = [file_list[i] for i in counts[2]]

    with open(os.path.join(dataset_path,"splitGroup.pt"), 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)




class imageAnalysis:
    def __init__(self):
        self.num = 0
        self.minsize = [1e10,1e10,1e10]
        self.maxsize = [0 , 0, 0]
        self.meanSize = np.zeros(3)
        self.minspace = [1e10,1e10,1e10]
        self.maxspace = [0, 0, 0]
        self.sameSpaceXYZ = True



    def getImageMessage(self,ct_path):
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        size = ct.GetSize()
        space = ct.GetSpacing()
        self.num += 1
        self.meanSize = 1.0*(self.meanSize *(self.num-1) + size)/self.num

        for i in range(3):
            self.minsize[i]= min(size[i],self.minsize[i])
            self.maxsize[i] = max(size[i], self.maxsize[i])
            self.minspace[i]= min(space[i],self.minspace[i])
            self.maxspace[i] = max(space[i], self.maxspace[i])

        if space[0]!=space[1] or space[1] != space[2]:
            self.sameSpaceXYZ = False



# 某一标签的范围
def getIntensityRange(seg_array, intensity):
    z = np.any(seg_array == intensity, axis=(1, 2))
    z_start_slice, z_end_slice = np.where(z)[0][[0, -1]]

    y = np.any(seg_array == intensity, axis=(0, 2))
    y_start_slice, y_end_slice = np.where(y)[0][[0, -1]]

    x = np.any(seg_array == intensity, axis=(0, 1))
    x_start_slice, x_end_slice = np.where(x)[0][[0, -1]]

    return [[z_start_slice, z_end_slice], [y_start_slice, y_end_slice], [x_start_slice, x_end_slice]]

#切除空气
def remove_slices_below_threshold(ct_array,seg_array, threshold=-500):
    # 检查在axis=(0, 2)和axis=(0, 1)上的切片
    mask_axis_0_2 = np.all(ct_array <= threshold, axis=(2))  # 沿着axis=(0, 2)移除
    mask_axis_0_1 = np.all(ct_array <= threshold, axis=(1))  # 沿着axis=(0, 1)移除

    # 使用掩码来过滤数组
    # 由于mask是二维的，我们需要确保两者都满足时才能移除切片
    final_mask = np.logical_and(mask_axis_0_2, mask_axis_0_1)
    ct_array_filtered = ct_array[~final_mask, :, :]  # 使用最终的掩码过滤
    seg_array_filtered = ceg_array[~final_mask, :, :]
    return ct_array_filtered,seg_array_filtered
#填充为立方体
def pad_to_cube(ct_array, seg_array):
    target_size = max(ct_array.shape)
    d, w, h = ct_array.shape
    pad_d = (target_size - d) // 2, target_size - d - (target_size - d) // 2
    pad_w = (target_size - w) // 2, target_size - w - (target_size - w) // 2
    pad_h = (target_size - h) // 2, target_size - h - (target_size - h) // 2

    ct_array_padded = np.pad(ct_array, (pad_d, pad_w, pad_h), mode='constant', constant_values=-1000)
    seg_array_padded = np.pad(seg_array, (pad_d, pad_w, pad_h), mode='constant', constant_values=-1000)

    return ct_array_padded, seg_array_padded , target_size


class CBCT_preprocess:
    def __init__(self, raw_dataset_path, fixed_dataset_path, args):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path
        # self.classes = args.n_labels  # 分割类别数（只分割牙与非牙，类别为2）
        self.upper = args.upper
        self.lower = args.lower
        self.expand_slice = args.expand_slice  # 轴向外侧扩张的slice数量
        self.size = args.min_slices  # 取样的slice数量
        self.xy_down_scale = args.xy_down_scale
        self.slice_down_scale = args.slice_down_scale
        #
        self.valid_rate = args.valid_rate
        self.args = args

    def fix_data(self):
        if not os.path.exists(self.fixed_path):  # 创建保存目录
            os.makedirs(self.fixed_path)
            os.makedirs(join(self.fixed_path, 'ct'))
            os.makedirs(join(self.fixed_path, 'seg'))

        file_list = os.listdir(join(self.raw_root_path, 'segmentation'))

        Numbers = len(file_list)
        simples = []
        ia = imageAnalysis()
        print('Total numbers of samples is :', Numbers)
        zsize = []
        for seg_file, i in zip(file_list, range(Numbers)):
            print("==== {} | {}/{} ====".format(seg_file, i + 1, Numbers))
            seg_path = os.path.join(self.raw_root_path, 'segmentation', seg_file)
            ct_path = os.path.join(self.raw_root_path, 'volume', seg_file.replace('segmentation','volume'))

            save_path = os.path.join(self.fixed_path,"{}.pickle".format(i))
            # new_ct, new_seg,zoomsize = self.resize(ct_path, seg_path)
            zoomsize = self.resize(ct_path, seg_path)
            zsize.append("{}\n".format(zoomsize))
            with open(save_path,'wb') as handle:
                pickle.dump([torch.from_numpy(zoomsize[0]),torch.from_numpy(zoomsize[1])], handle, protocol=pickle.HIGHEST_PROTOCOL)



    def resize(self, ct_path, seg_path):
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        print("Ori shape:", ct_array.shape, seg_array.shape)

        # 将灰度值在阈值之外的截断掉
        ct_s,seg_s = remove_slices_below_threshold(ct_array,seg_array, -500)

        # 截取保留区域
        ct_padded,seg_padded,max_size = pad_to_cube(ct_s,seg_s)
        zoom_size = 512/max_size
        # return zoom_size
        ct_array = ndimage.zoom(ct_padded, (zoom_size, zoom_size, zoom_size), order=3)
        seg_array = ndimage.zoom(seg_padded, (zoom_size, zoom_size, zoom_size), order=0)

        print("resize shape:", ct_array.shape, seg_array.shape)
        return ct_array, seg_array, zoom_size


if __name__ == '__main__':
    raw_dataset_path = '/home/yisiinfo/cyj/cj/segmentation_teeth_3d/CBCT_data'
    fixed_dataset_path = '/home/yisiinfo/cyj/cj/segmentation_teeth_3d/CBCT_data2'

    args = config.args
    tool = CBCT_preprocess(raw_dataset_path, fixed_dataset_path, args)

    # tool.fix_data()  # 对原始图像进行修剪并保存
    # splitDataset(fixed_dataset_path)

    img_path = ""
    seg_path = ""
    save_img_path = ""
    save_seg_path = ""
    ct_array, seg_array, zoom_size = CBCT_preprocess.resize(img_path,seg_path)
    sitk.WriteImage(ct_array, save_img_path)
    sitk.WriteImage(seg_array,save_seg_path)



