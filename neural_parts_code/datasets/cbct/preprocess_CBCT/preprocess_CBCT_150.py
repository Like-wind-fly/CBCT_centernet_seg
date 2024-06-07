# 这个预处理只是将nii格式的文件，转为容易读取写的.pickle文件
# 会将图像的长宽高统一步长，删除掉多余的部分，缩放为统一的大小，对于小于标准的大小的图像进行
# 填充。
#具体参数定义在config.py
import numpy as np
import os
import SimpleITK as sitk
from scipy import ndimage
from os.path import join
import math
import torch
import pickle
import gc

class DataModes:
    TRAINING = 'training'
    VALIDATION = 'validation'
    TESTING = 'testing'
    ALL = 'all'
    def __init__(self):
        dataset_splits = [DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]

def splitDataset(dataset_path):
    file_list = os.listdir(dataset_path)

    file_list = [int(fname[:-7]) for fname in file_list if "pickle" in fname]
    file_list.sort()
    file_list = ["{}.pickle".format(fname) for fname in file_list]
    file_list = [os.path.join(dataset_path, fname) for fname in file_list]
    # file_list = file_list[:97]
    toothNum = len(file_list)

    print('Total numbers of samples is :', toothNum)
    np.random.seed(0)
    perm = np.random.permutation(toothNum)
    counts = [perm[:int(toothNum * 0.6)], perm[int(toothNum * 0.6):int(toothNum * 0.8)],perm[int(toothNum * 0.8):int(toothNum)]]

    dataset = {}

    dataset[DataModes.TRAINING] = [file_list[i] for i in counts[0]]
    dataset[DataModes.VALIDATION] = [file_list[i] for i in counts[1]]
    dataset[DataModes.TESTING] = [file_list[i] for i in counts[2]]

    with open(os.path.join(dataset_path,"splitGroup_122.pt"), 'wb') as handle:
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

#tensor_list 里的 D,W,H = tensor.shape
def roiCrop(tensor_list,center,sizes,pad_vaule):
    tensor_shape = tensor_list.shape

    crop = []
    for i in range(3):
        crop.append([center[i] - math.floor(sizes[i] / 2.0),center[i] + math.ceil(sizes[i]  / 2.0)])

    crop = torch.tensor(crop).long()
    crop_over = crop.clone()
    crop_over[crop_over<0] = 0

    tensor_list = tensor_list[crop_over[0][0]:crop_over[0][1], crop_over[1][0]:crop_over[1][1], crop_over[2][0]:crop_over[2][1]]

    padding = []
    for i in range(3):
        left = 0
        right = 0

        if crop[i][0] <0:
            left -= crop[i][0]
        if crop[i][1] > tensor_shape[i]:
            right = crop[i][1] - tensor_shape[i]

        padding.append(right)
        padding.append(left)

    padding.reverse()
    pad = torch.nn.ConstantPad3d(padding, value=pad_vaule)
    tensor_list = pad(tensor_list[None,None])[0,0]
    return tensor_list

# 某一标签的范围
def getIntensityRange(seg_array, intensity):
    z = np.any(seg_array == intensity, axis=(1, 2))
    z_start_slice, z_end_slice = np.where(z)[0][[0, -1]]

    y = np.any(seg_array == intensity, axis=(0, 2))
    y_start_slice, y_end_slice = np.where(y)[0][[0, -1]]

    x = np.any(seg_array == intensity, axis=(0, 1))
    x_start_slice, x_end_slice = np.where(x)[0][[0, -1]]

    return [[z_start_slice, z_end_slice], [y_start_slice, y_end_slice], [x_start_slice, x_end_slice]]


from matplotlib import pyplot as plt
class CBCT_preprocess:
    def __init__(self, raw_dataset_path, fixed_dataset_path, args):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path
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
        file_list.sort();
        Numbers = len(file_list)
        print('Total numbers of samples is :', Numbers)
        zsize = []

        for seg_file, i in zip(file_list, range(Numbers)):
            if 120<=i:
                print("==== {} | {}/{} ====".format(seg_file, i + 1, Numbers))
                seg_path = os.path.join(self.raw_root_path, 'segmentation', seg_file)
                ct_path = os.path.join(self.raw_root_path, 'volume', seg_file)

                save_path = os.path.join(self.fixed_path,"{}.pickle".format(i))
                new_ct, new_seg, zoom_size = self.resize(ct_path, seg_path)
                if zoom_size == 0:
                    print(ct_path)
                    continue

                zsize.append("{}\n".format(zsize))
                # 单层列表写入文件

                with open(save_path,'wb') as handle:
                    pickle.dump([torch.from_numpy(new_ct),torch.from_numpy(new_seg)], handle, protocol=pickle.HIGHEST_PROTOCOL)
                ct_img = sitk.GetImageFromArray(new_ct)
                seg_img = sitk.GetImageFromArray(new_seg)
                sitk.WriteImage(seg_img, os.path.join(self.fixed_path,"seg/{}.nii".format(i)))
                sitk.WriteImage(ct_img, os.path.join(self.fixed_path,"ct/{}.nii".format(i)))

                del ct_img
                del seg_img
                del new_ct
                del new_seg
                gc.collect()

        # # 单层列表写入文件
        # with open(os.path.join(self.fixed_path,"zoomsize.txt"), "w") as f:
        #     f.writelines(zsize)


    def resize(self, ct_path, seg_path):
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        print("Ori shape:", ct_array.shape, seg_array.shape)

        # 将灰度值在阈值之外的截断掉
        ct_array[ct_array > self.upper] = self.upper
        ct_array[ct_array < self.lower] = self.lower
        ct_array = ct_array - self.lower
        ct_array = ct_array / (self.upper-self.lower)

        # 找到牙齿区域开始和结束的slice，并各向外扩张
        print("Ori shape:", ct_array.shape, seg_array.shape)
        pro_array = seg_array.copy()
        pro_array[pro_array>0] = 1
        if pro_array.sum() < 1000:
            return None,None,0

        xyz_range = getIntensityRange(pro_array, 1)

        box_center = []
        box_size = []
        for i in range(3):
            start_slice, end_slice = xyz_range[i]
            # 两个方向上各扩张个slice
            if start_slice - self.expand_slice < 0:
                start_slice = 0
            else:
                start_slice -= self.expand_slice

            if end_slice + self.expand_slice >= pro_array.shape[i]:
                end_slice = pro_array.shape[i] - 1
            else:
                end_slice += self.expand_slice
            xyz_range[i] = [start_slice, end_slice]
            print("Cut out range:{}    ".format(i)+str(start_slice) + '--' + str(end_slice))
            print(end_slice - start_slice)
            box_center.append((end_slice + start_slice)/2)
            box_size.append(end_slice - start_slice)

        # 截取保留区域
        crop_size = [128,256,256]
        crop_size[0] = int(np.ceil(box_size[0]/32)*32)

        ct_array = roiCrop(torch.tensor(ct_array),box_center,crop_size,0.01).numpy()
        seg_array = roiCrop(torch.tensor(seg_array),box_center,crop_size,0).numpy()


        # plt.imshow(ct_array[70, :, :],cmap="gray")
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(seg_array[70, :, :],cmap="gray")
        # plt.colorbar()
        # plt.show()

        print("resize shape:", ct_array.shape, seg_array.shape)
        return ct_array, seg_array, 1


import config
if __name__ == '__main__':
    raw_dataset_path   = '/data/CBCT150/raw_data'
    fixed_dataset_path = '/data/CBCT150/CBCT_Inpainting'

    # args = config.args
    # tool = CBCT_preprocess(raw_dataset_path, fixed_dataset_path, args)
    #
    # tool.fix_data()  # 对原始图像进行修剪并保存
    splitDataset(fixed_dataset_path)