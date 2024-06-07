import random
import pickle
import torch
from torchvision.ops import masks_to_boxes

from torch.utils.data import Dataset as dataset
from torch.utils.data import DataLoader
from neural_parts_code.datasets.cbct.common import centerCrop3d,heatmap
from neural_parts_code.datasets.transform import  stns
from neural_parts_code.datasets.cbct.config import args
from neural_parts_code.models.boxDecode import roiCrop
import numpy as np
import math
import random
from neural_parts_code.datasets.cbct.common import save_img
class DataModes:
    TRAINING = 'training'
    VALIDATION = 'validation'
    TESTING = 'testing'
    ALL = 'all'
    def __init__(self):
        dataset_splits = [DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]

class CBCTDataset(dataset):
    def __init__(self, data, mode, conf):
        self.data = data
        self.mode = mode
        self.configs = conf
        self.readFromMen = False
        # self.cropTeeth = conf["data"].get("cropTeeth",False)

    def __len__(self):
        # return min(len(self.data),10)
        return len(self.data)

    def readImageFromMen(self):
        dataset = []

        for idx in range(self.__len__()):
            samplepath = self.data[idx]
            with open(samplepath, 'rb') as handle:
                ct, seg = pickle.load(handle)
            dataset.append([ct, seg])
            print(idx)

        self.data = dataset
        self.readFromMen = True

    def __getitem__(self, idx):
        if self.readFromMen:
            ct, seg = self.data[idx]
        else:
            samplepath = self.data[idx]
            with open(samplepath, 'rb') as handle:
                ct,seg = pickle.load(handle)
        with torch.no_grad():
            ct = ct.float().cuda()
            seg = seg.float().cuda()

            if self.mode != DataModes.TRAINING:
                test_crop_size = self.configs.test_crop_size
                ct = centerCrop3d(ct[None], test_crop_size)[0] #裁剪
                seg = centerCrop3d(seg[None], test_crop_size)[0]


            quad_ct, quad_seg = augmentation(ct,seg, self.mode, self.configs)  #数据增强后的
            #1是4个相限的相限标签，2是分区的ct，3是分区的seg标签，4相限的中心点
            quad_seg, identify_ct, identify_seg, index = process(ct,seg,quad_seg,self.mode)   #数据处理原始CT和标签，增强后的标签得到对应增强的分割图、分区 CT 扫描和分割图、以及象限中心点。

            quad = torch.range(0,3).long().cuda() #随机选取两个象限
            if self.mode == DataModes.TRAINING:  # if training do augmentation
                quad = torch.tensor(np.random.permutation(4)).cuda()
                quad = quad[:2]

            identify_ct = identify_ct[quad]  #拿分区CT
            identify_seg = identify_seg[quad] #某个分区的分割
            #将分割图中的标签从 11-18 和 21-28 转换为 1-8。
            seg = seg.int() #分割标签
            for i in range(1, 5):
                for j in range(1, 9):
                    # print("{} <---- {} ".format((i - 1) * 8 + j, (i * 10 + j)))
                    seg[seg == (i * 10 + j)] = (i - 1) * 8 + j

            ret = {
                   'identify_ct': identify_ct, #识别 CT 扫描（分区 CT 扫描）
                    'quad_ct': quad_ct[None], #象限 CT 扫描
                   'identify_seg': identify_seg,#识别分割图（分区分割图）
                   'quad_seg': quad_seg[None], #象限分割图
                   'quad_id' : quad, #象限 ID
                   'center': index, #象限中心点
                    'ct': ct[None], #原始 CT 扫描
                    'seg': seg[None] #原始分割图
                   }
            return ret




def process(ct,seg,quad_seg,mode):
    # 将牙齿标记为4象限与8标号
    quad_seg_4 = torch.floor(quad_seg/10)  #标签对10除得到分区号，分成4个区
    #seg的标签是background为0，其他为标签11-18，21-28……所以按照前面的数值定义象限。
    seg_4 = torch.floor(seg/10) #和quad_seg_4一致，是区号
    seg_8 = seg%10 #对标签求余得到分区牙号

    # 将4象限的label转为4通道的mask
    one_map = torch.stack([seg_4 == id for id in range(1,5)],dim=0)  #四通道张量，每个通道一个象限

    #根据mask求包围框，每个象限的边界范围
    ETMaskDW = one_map.sum(dim = -1)>0 #二进制，宽
    ETMaskDH = one_map.sum(dim=  2)>0

    rangeDW = masks_to_boxes(ETMaskDW) #张量，分区象限边界框
    rangeDH = masks_to_boxes(ETMaskDH)
    #这个代码有问题
    rangeDWH = torch.cat([rangeDW[:,[1,3]],rangeDW[:,[0,2]],rangeDH[:,[0,2]]],dim= 1) #

    dwh = torch.stack([(rangeDWH[:,i * 2 + 1] - rangeDWH[:,i * 2]) for i in range(3)], dim=1) #象限长宽深
    #调整空间的大小，大概率需要调整
    dwh[:, 0] = 96
    dwh[:, 1] = 256
    dwh[:, 2] = 128
    index = torch.stack([(rangeDWH[:,i * 2 + 1] + rangeDWH[:,i * 2]) for i in range(3)], dim=1) #象限中心点
    index = (index/2).int()
    if mode == DataModes.TRAINING: #随机平移
        shift = torch.randint(low = -5,high=5,size = (1,3) ).cuda()
        index+= shift

    part_ct_list = []
    part_seg_8_list = []

    for i in range(4):
        center = index[i] #中心
        sizes  = dwh[i]
        mask = one_map[i]

        part_ct, part_seg_8, part_mask = roiCrop([ct,seg_8,mask],center,sizes) #获取roi
        part_seg_8 =  part_seg_8 * part_mask #和8分区牙和掩码求交

        part_ct_list.append(part_ct)
        part_seg_8_list.append(part_seg_8)

    part_ct_list = torch.stack(part_ct_list, dim=0) #是一个列表，其中包含每个象限的 CT 扫描裁剪。
    part_seg_8_list = torch.stack(part_seg_8_list, dim=0) ##是一个列表，其中包含每个象限的分割图裁剪。

    return quad_seg_4,part_ct_list,part_seg_8_list,index #四象限分割，CT扫描裁剪列表，分割裁剪列表，象限分区中心


def augmentation(ct_tensor,seg_tensor, mode, configs):
    # augmentation done only during training
    f = configs.scale_factor
    train_crop_size = configs.train_crop_size
    test_crop_size = configs.test_crop_size

    if mode == DataModes.TRAINING:  # if training do augmentation
        ct_tensor,seg_tensor = rangeCrop([ct_tensor,seg_tensor], train_crop_size)
        # 旋转 ，xyz 随机颠倒
        # angle = np.pi / 6  # 45 degrees
        # theta_rotate = stns.rotate(angle)
        # 不颠倒
        theta_rotate = torch.zeros(4,4)
        for i in range(4):
            theta_rotate[i, i] = 1.0
        # 不平移

        # shift = torch.tensor([0.2,0.2,0.2])

        # Define the maximum shift range (in pixels)
        numbers = [-0.2, -0.1, 0, 0.1, 0.2]
        # Generate a random shift vector within the range [-max_shift, max_shift]
        shift = torch.tensor([
            random.choice(numbers),
            random.choice(numbers),
            random.choice(numbers),
        ])

        theta_shift = stns.shift(shift)

        # 缩放
        scale = 1.0 - 2 * f * (torch.rand(1) - 0.5)
        # scale = 1.0
        theta_scale = stns.scale(scale)


        theta = theta_rotate @ theta_shift @ theta_scale
        ct_tensor = stns.transform(theta, ct_tensor[None])
        seg_tensor = stns.transform(theta, seg_tensor[None] ,sample_mode = 'nearest')
        save_img(ct_tensor.cpu().numpy(),'ct_tensor.nii')
        save_img(seg_tensor.cpu().numpy(), 'seg_tensor.nii')
    else:
        ct_tensor = centerCrop3d(ct_tensor[None], test_crop_size)
        seg_tensor = centerCrop3d(seg_tensor[None], test_crop_size)

    return ct_tensor[0], seg_tensor[0] #看是否是训练模型，进行增强或者裁剪


def rangeCrop(tensor_list,sizes):
    shape = tensor_list[0].shape
    crop = torch.tensor(getCropRange(shape, sizes))
    tensor_list = [x[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], crop[2][0]:crop[2][1]] for x in tensor_list]

    padding = []
    for i in range(3):
        left = 0
        right = 0
        if sizes[i] > shape[i]:
            left = math.floor((sizes[i] - shape[i]) / 2)
            right = math.ceil((sizes[i] - shape[i]) / 2)
        padding.append(left)
        padding.append(right)
    padding.reverse()
    pad = torch.nn.ConstantPad3d(padding, value=0)
    tensor_list = [pad(x) for x in tensor_list]
    return tensor_list

def getCropRange(shape,sizes = [128,128,128]):
    crop = []
    for i in range(3):
        if shape[i]<sizes[i]:
            crop.append((0,shape[i]))
        else:
            size  = sizes[i]
            min = size/2
            max = shape[i]-size/2
            center = random.randint(min,max)
            crop.append((int(center-size/2),int(center+size/2)))
    return crop

class BuildCBCTSGADataSet():
    def __init__(self,conf):
        self.data_path = conf["data"]["dataset_directory"]
        args.test_crop_size = conf["data"].get("test_crop_size",args.test_crop_size)
        args.train_crop_size = conf["data"].get("train_crop_size", args.train_crop_size)
        self.config = args

    def quick_load_data(self):
        data_path = self.data_path

        with open(data_path, 'rb') as handle:
            datasets = pickle.load(handle)

        down_data = {}
        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING, DataModes.VALIDATION]):
            data = datasets[datamode]
            down_data[datamode] = CBCTDataset(data, datamode, self.config) ##这里得到下采样处理好的数据

        return down_data



if __name__ == "__main__":
    from common import save_img
    from scripts.utils import load_config
    from neural_parts_code.datasets import build_datasets
    config = load_config("/home/zhangzechu/workspace/neral-parts-CBCT/config/sga_net.yaml")
    torch.cuda.set_device(int(config["network"]["cuda"]))
    datasets = build_datasets(config)

    # 定义数据加载
    for i, dataset in enumerate(datasets[2:]):
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for j,data in enumerate(loader):
            quad_ct = data["quad_ct"][0,0]
            quad_seg = data["quad_seg"][0,0]
            identify_ct = data["identify_ct"][0]
            identify_seg = data["identify_seg"][0]

            save_img(quad_ct.cpu().numpy(),"./nii/quad_ct_{}_{}.nii".format(i,j))
            save_img(quad_seg.int().cpu().numpy(),"./nii/quad_seg_{}_{}.nii".format(i,j))

            for k in range(4):
                part_ct = identify_ct[k]
                part_seg = identify_seg[k]
                save_img(part_ct.cpu().numpy(), "./nii/part_ct_{}_{}_{}.nii".format(i, j, k))
                save_img(part_seg.int().cpu().numpy(), "./nii/part_seg_{}_{}_{}.nii".format(i, j, k))

            break



