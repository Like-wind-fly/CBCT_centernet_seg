import os.path
import random
import pickle
import torch
from torchvision.ops import masks_to_boxes

from torch.utils.data import Dataset as dataset
from torch.utils.data import DataLoader
from neural_parts_code.datasets.cbct.common import centerCrop3d,heatmap
from neural_parts_code.datasets.transform import  stns
from neural_parts_code.datasets.cbct.config import args
from neural_parts_code.models.boxDecode import cropInput
import numpy as np
import math

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
            ct, seg = augmentation(ct,seg, self.mode, self.configs)
            seg,heatmap_tensor,heatmap32_tensor,reg_mask,index_int,reg,dwh = process(seg,self.configs.down_ratio,self.configs.kernel_size, self.configs.sigma, self.configs.classNum)

            # ret = [ct[None],       # cbct的影像      1, D，H, W
            #        seg[None],      # 标注的分割的图    1, D，H, W
            #        heatmap_tensor,  # heatmap         c, D, H, W
            #        reg_mask,  # 用来标注到底出现了多少实例，   32
            #        index_int,  # 用来标注每一个实例的中心     32,3
            #        reg,        # 下采样导致的偏差            32,3
            #        dwh           # dept high weight       32,3
            #        ]

            ret = {
                   'ct': ct[None],
                   'seg': seg[None],
                   'heatmap': heatmap_tensor,
                   'heatmap32': heatmap32_tensor,
                   'reg_mask': reg_mask,
                   'index_int':index_int,
                   'reg': reg,
                   'dwh': dwh
                   }
            return ret



def process(seg,down_ratio,kernel_size = 3 ,sigma=1,classNum = 1):
    # 将八进制的mask标注转为十进制
    D, W, H = seg.shape
    one_map_id = []
    for i in range(1, 5):
        for j in range(1, 9):
            one_map_id.append(i * 10 + j)
    # one_map_id = torch.tensor(one_map_id).cuda().int()

    # 11,12,13  -> 9,10,11  八 -》 十
    one_map = torch.stack([seg == id for id in one_map_id],dim=0)
    # one_map = seg == one_map_id[:, None, None, None]

    # value = torch.arange(1, 33).cuda()[:, None, None, None].repeat(1, D, W, H)
    seg = one_map[0].int()
    for i in range(2,33):
        seg+= one_map[i-1] * i

    # seg = (one_map * value).sum(dim=0)
    reg_mask = one_map.sum(dim=[1,2,3])>25

    existTeeth = one_map[reg_mask]

    #根据mask求包围框
    ETMaskDW = existTeeth.sum(dim = -1)>0
    ETMaskDH = existTeeth.sum(dim=  2)>0

    rangeDW = masks_to_boxes(ETMaskDW)
    rangeDH = masks_to_boxes(ETMaskDH)
    rangeDWH = torch.cat([rangeDW[:,[1,3]],rangeDW[:,[0,2]],rangeDH[:,[0,2]]],dim= 1)
    dwh = torch.stack([(rangeDWH[:,i * 2 + 1] - rangeDWH[:,i * 2]) for i in range(3)], dim=1)
    index = torch.stack([(rangeDWH[:,i * 2 + 1] + rangeDWH[:,i * 2]) for i in range(3)], dim=1)
    index = (index/2).int()
    index = 1.0*index/down_ratio
    index_int = index.int()
    reg = index - index_int
    D,W,H = seg.shape
    shape = torch.Size([D//down_ratio,W//down_ratio,H//down_ratio])

    # 找到各个牙齿的中心，根据牙齿与中心的远近程度，设置概率值，用0-255标识
    heatmap_tensor = heatmap(shape,index_int,kernel_size = kernel_size ,sigma= sigma)
    b, _, d, w, h = heatmap_tensor.shape

    temp = torch.zeros(32, d, w, h).cuda()
    temp[reg_mask] = heatmap_tensor.squeeze(dim=1)
    heatmap32_tensor = temp


    if classNum == 1:

        if b == 0:
            heatmap_tensor = torch.zeros(1,d,w,h).cuda()
        else:
            heatmap_tensor,_ = heatmap_tensor.max(dim = 0)
    elif classNum == 8:
        heatlist = [[] for i in range(8)]
        id =0
        for j in range(reg_mask):
            if reg_mask[j]:
                heatlist[j%8].append(heatmap_tensor[id])
                id += 1

        for i in range(8):
            if len(heatlist[i]) ==0:
                heatlist[i] = torch.zeros(shape).cuda()
            elif len(heatlist[i]) ==1:
                heatlist[i] = heatlist[i][0]
            elif len(heatlist[i]) > 1:
                heatlist[i],_ = torch.stack(heatlist[i],dim=0).max(dim=0)
        heatmap_tensor = torch.stack(heatlist,dim=0)

    elif classNum == 32:
        temp = torch.zeros(32,d,w,h).cuda()
        temp[reg_mask] = heatmap_tensor.squeeze(dim=1)
        heatmap_tensor = temp


    temp = torch.zeros(32,3).cuda().int()
    temp[reg_mask] = index_int
    index_int = temp

    temp = torch.zeros(32,3).cuda()
    temp[reg_mask] = reg
    reg = temp

    temp = torch.zeros(32,3).cuda().int()
    temp[reg_mask] = dwh.int()
    dwh = temp

    # torch.cuda.empty_cache()
    return seg,heatmap_tensor,heatmap32_tensor,reg_mask,index_int,reg,dwh


def augmentation(ct_tensor,seg_tensor, mode, configs):
    # augmentation done only during training
    f = configs.scale_factor
    train_crop_size = configs.train_crop_size
    test_crop_size = configs.test_crop_size

    if mode == DataModes.TRAINING:  # if training do augmentation
        ct_tensor,seg_tensor = rangeCrop([ct_tensor,seg_tensor], train_crop_size)
        # 旋转 ，xyz 随机颠倒
        theta_rotate = torch.zeros(4,4)
        # for i in range(4):
        #     theta_rotate[i, i] = 1.0
        perm = torch.tensor(np.random.permutation(3))
        sign = (torch.rand(3) - 0.5)
        sign[sign>0] =  1
        sign[sign<0] = -1
        sign[:] = 1

        for i in range(3):
            theta_rotate[i, perm[i]] = sign[i]
        theta_rotate[3, 3] = 1.0

        shift = torch.tensor([0,0,0])
        theta_shift = stns.shift(shift)

        scale = 1.0 - 2 * f * (torch.rand(1) - 0.5)
        theta_scale = stns.scale(scale)

        theta = theta_rotate @ theta_shift @ theta_scale

        ct_tensor = stns.transform(theta, ct_tensor[None])
        seg_tensor = stns.transform(theta, seg_tensor[None] ,sample_mode = 'nearest')

    else:
        ct_tensor = centerCrop3d(ct_tensor[None], test_crop_size)
        seg_tensor = centerCrop3d(seg_tensor[None], test_crop_size)

    return ct_tensor[0], seg_tensor[0]


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

class BuildCBCTDataSet():
    def __init__(self,conf):
        self.data_path = conf["data"]["dataset_directory"]
        used_layer = conf["town"]["used_layer"]
        args.kernel_size = conf["data"]["kernel_size"]
        args.sigma = conf["data"]["sigma"]
        args.down_ratio = 2**used_layer

        args.test_crop_size = conf["data"].get("test_crop_size",args.test_crop_size)
        args.train_crop_size = conf["data"].get("train_crop_size", args.train_crop_size)
        args.classNum = conf["data"].get("classNum", args.classNum)
        self.config = args

    def quick_load_data(self):
        data_path = self.data_path

        with open(data_path, 'rb') as handle:
            datasets = pickle.load(handle)

        down_data = {}
        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING, DataModes.VALIDATION]):
            data = datasets[datamode]
            down_data[datamode] = CBCTDataset(data, datamode, self.config)

        return down_data



if __name__ == "__main__":
    from common import save_img
    from scripts.utils import load_config
    from neural_parts_code.datasets import build_datasets
    config = load_config("/home/zhangzechu/workspace/neral-parts-CBCT/config/centernet.yaml")
    torch.cuda.set_device(int(config["network"]["cuda"]))
    datasets = build_datasets(config)
    save_path = "/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/gt"
    # 定义数据加载
    for i, dataset in enumerate(datasets[2:]):
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for j,data in enumerate(loader):
            print(j)
            ct = data["ct"][0,0]
            seg = data["seg"][0,0]

            # heatmap_tensor = data[2][0, 0]
            # reg_mask = data[3][0]
            # index_int = data[4][0]
            # reg = data[5][0]
            # dwh = data[6][0]

            save_img(ct.cpu().numpy(),os.path.join(save_path,"ct_{}.nii".format(j)))
            save_img(seg.int().cpu().numpy(),os.path.join(save_path,"volume_{}.nii".format(j)))
            torch.save(seg,os.path.join(save_path,"volume_{}.pt".format(j)))
            print("seg_{}".format(j))
            print(seg.shape)
            # save_img(heatmap_tensor.cpu().numpy(),"heatmap.nii")
            # box = ct.clone()
            # for k in range(len(reg_mask)):
            #     exist_ = reg_mask[k]
            #     index_ = index_int[k]
            #     reg_  = reg[k]
            #     dwh_ = dwh[k]
            #     down_ratio = args.down_ratio
            #     if exist_:
            #         center = (index_+reg_) * down_ratio
            #         a = (center - dwh_/2).int()
            #         b = (center + dwh_ / 2).int()
            #         box[a[0]:b[0],a[1]:b[1],a[2]:b[2]] = 1
            #
            # save_img(box.cpu().numpy(), "box.nii")




