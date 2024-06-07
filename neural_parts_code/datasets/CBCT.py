import time

import numpy as np
import sys
import os
from os.path import join

import pickle

import igl
import SimpleITK as sitk
from skimage import measure
import torch.nn.functional as F
from torch.utils.data import Dataset as dataset

import math
import torch
import argparse
from scipy.spatial.transform import Rotation as R


class Sample:
    def __init__(self, x, y,verts = None,faces = None, hot_map = None):
        self.x = x
        self.y = y
        self.verts = verts
        self.faces = faces
        self.hot_map = hot_map

    def cpu(self):
        if self.x != None:
            self.x = self.x.cpu()
        if self.y!= None:
            self.y = self.y.cpu()
        if self.verts!= None:
            self.verts = self.verts.cpu()
        if self.faces!= None:
            self.faces = self.faces.cpu()
        if self.hot_map!= None:
            self.hot_map = self.hot_map.cpu()
        return self

    def cuda(self):
        if self.x != None:
            self.x = self.x.cuda()
        if self.y!= None:
            self.y = self.y.cuda()
        if self.verts!= None:
            self.verts = self.verts.cuda()
        if self.faces!= None:
            self.faces = self.faces.cuda()
        if self.hot_map!= None:
            self.hot_map = self.hot_map.cuda()
        return self

class SamplePlus:
    def __init__(self, sample , y_outer=None, x_super_res=None, y_super_res=None, shape=None):
        #from simple
        self.x = sample.x
        self.y = sample.y
        self.verts = sample.verts
        self.faces = sample.faces
        self.hot_map = sample.hot_map
        # addition
        self.y_outer = y_outer.cpu()
        self.x_super_res = x_super_res
        self.y_super_res = y_super_res
        self.shape = shape

    def cpu(self):
        if self.x != None:
            self.x = self.x.cpu()
        if self.y!= None:
            self.y = self.y.cpu()
        if self.verts!= None:
            self.verts = self.verts.cpu()
        if self.faces!= None:
            self.faces = self.faces.cpu()
        if self.hot_map!= None:
            self.hot_map = self.hot_map.cpu()
        if self.y_outer!= None:
            self.y_outer = self.y_outer.cpu()
        if self.x_super_res!= None:
            self.x_super_res = self.x_super_res.cpu()
        if self.y_super_res!= None:
            self.y_super_res = self.y_super_res.cpu()
        return self

    def smallSize(self):
        if self.x != None:
            self.x = self.x.to(torch.float32)
        if self.y!= None:
            self.y = self.y.to(torch.int16)
        if self.verts!= None:
            self.verts = self.verts.to(torch.float32)
        if self.faces!= None:
            self.faces = self.faces.to(torch.int16)
        if self.hot_map!= None:
            self.hot_map = self.hot_map.to(torch.float32)
        if self.y_outer!= None:
            self.y_outer = self.y_outer.to(torch.int16)
        if self.x_super_res!= None:
            self.x_super_res = self.x_super_res.to(torch.float32)
        if self.y_super_res!= None:
            self.y_super_res = self.y_super_res.to(torch.int16)
        return self


class CBCTDataset(dataset):
    def __init__(self, data, cfg, mode): 
        self.data = data
        self.cfg = cfg
        self.mode = mode

    def __len__(self):
        # return 64
        return len(self.data)

    def __getitem__(self, idx):
        samplepath = self.data[idx]
        item = None
        with open(samplepath, 'rb') as handle:
            item = pickle.load(handle)

        return get_item(item, self.mode, self.cfg) 

  

class CBCT():
    def __init__(self):
        parser = argparse.ArgumentParser(description='CBCT management')
        # Preprocess parameters
        parser.add_argument('--n_labels', type=int, default=2, help='number of classes')  # 分割类别数（只分割牙与非牙，类别为2）
        parser.add_argument('--upper', type=int, default=2500, help='')
        parser.add_argument('--lower', type=int, default=-500, help='')
        parser.add_argument('--median', type=int, default=1000, help='')

        parser.add_argument('--norm_factor', type=float, default=1500, help='')
        parser.add_argument('--expand_slice', type=int, default=16, help='')
        parser.add_argument('--min_slices', type=int, default=48, help='')
        parser.add_argument('--xy_down_scale', type=float, default=1, help='')
        parser.add_argument('--slice_down_scale', type=float, default=1, help='')

        # data in/out and dataset

        # train
        parser.add_argument('--crop_size', type=list, default=[96, 192, 192])
        parser.add_argument('--tooth_crop_size', type=list, default=[64, 64, 64])
        parser.add_argument('--val_crop_max_size', type=int, default=128)

        # test
        parser.add_argument('--test_cut_size', type=int, default=48, help='size of sliding window')
        parser.add_argument('--test_cut_stride', type=int, default=24, help='stride of sliding window')
        parser.add_argument('--postprocess', type=bool, default=False, help='post process')

        self.args = parser.parse_args()

    def pick_surface_points(self, y_outer, point_count):
        idxs = torch.nonzero(y_outer)
        perm = torch.randperm(len(idxs))

        y_outer = y_outer * 0
        idxs = idxs[perm[:point_count]]
        y_outer[idxs[:,0], idxs[:,1], idxs[:,2]] = 1
        return y_outer



    # # 先用 64*64*64 分辨率做一下实验
    def quick_load_data(self, cfg):
        down_sample_shape = cfg.patch_shape
        data_root = cfg.dataset_path
        down_data = {}

        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING, DataModes.VALIDATION]):
            dirPath = join(data_root, datamode)
            file_list = os.listdir(dirPath)
            file_list = ["{}/{}".format(dirPath,filename) for filename in file_list  if 'pickle' in filename]
            down_data[datamode] = CBCTDataset(file_list, cfg, datamode)
        return down_data



def getGrid(D,H,W):
    # we resample such that 1 pixel is 1 mm in x,y and z directiions
    base_grid = torch.zeros((1, D, H, W, 3)).cuda()
    w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
    h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
    d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)
    base_grid[:, :, :, :, 0] = w_points
    base_grid[:, :, :, :, 1] = h_points
    base_grid[:, :, :, :, 2] = d_points

    return base_grid

def sample_outer_surface_in_voxel(volume):
    # outer surface
    a = F.max_pool3d(volume[None,None].float(), kernel_size=(3,1,1), stride=1, padding=(1, 0, 0))[0]
    b = F.max_pool3d(volume[None,None].float(), kernel_size=(1,3, 1), stride=1, padding=(0, 1, 0))[0]
    c = F.max_pool3d(volume[None,None].float(), kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))[0]
    border, _ = torch.max(torch.cat([a,b,c],dim=0),dim=0)
    surface = border - volume.float()
    return surface.to(torch.int16)



# 把边界上的像素设为0，这样 matching_cubes不会出现空洞
def clean_border_pixels(image, gap):
    '''
    :param image:
    :param gap:
    :return:
    '''
    assert len(image.shape) == 3, "input should be 3 dim"

    D, H, W = image.shape
    y_ = image.clone()
    y_[:gap] = 0
    y_[:, :gap] = 0
    y_[:, :, :gap] = 0
    y_[D - gap:] = 0
    y_[:, H - gap] = 0
    y_[:, :, W - gap] = 0

    return y_

def normalize_vertices(vertices, shape):
    assert len(vertices.shape) == 2 and len(shape.shape) == 2, "Inputs must be 2 dim"
    assert shape.shape[0] == 1, "first dim of shape should be length 1"

    return 2*(vertices/(torch.max(shape)-1) - 0.5)


def get_item(item, mode, config):
    x = item.x.cuda()[None]
    y = item.y.cuda()[None]
    y_outer = item.y_outer.cuda()[None]
    hot_map = item.hot_map.cuda()[None]
    hot_map[hot_map<0.98] = 0

    verts = item.verts.cuda()[None]
    faces = item.faces.cuda()[None]
    shape = item.shape

    # augmentation done only during training
    if mode == DataModes.TRAINING:  # if training do augmentation
        # 旋转
        theta_rotate = torch.eye(4)
        a = R.random().as_matrix()
        for i in range(3):
            for j in range(3):
                theta_rotate[i, j] = a[i, j]

        shift = torch.tensor(
            [d / (D // 2) for d, D in zip(2 * (torch.rand(3) - 0.5) * config.augmentation_shift_range, y[0].shape)])
        theta_shift = stns.shift(shift)

        f = 0.1
        scale = 1.0 - 2 * f * (torch.rand(1) - 0.5)
        theta_scale = stns.scale(scale)

        theta = theta_rotate @ theta_shift @ theta_scale

        x, y ,hot_map ,y_outer = stns.transform(theta, x, y,hot_map,y_outer)
        verts = stns.transformVert(theta, verts[0])[None]

    # vertices_mc_all，faces_mc_all 加一维
    surface_points_normalized_all = None
    for i in range(1, config.num_classes):


        # 打乱
        point_count = 1536
        # perm = torch.randperm(verts.shape[1])
        # surface_points_normalized_all = verts[:, perm[:min(len(perm),point_count)]].cuda()

        surface_verts = torch.nonzero(y_outer[0]).float()
        surface_verts = torch.flip(surface_verts, dims=[1]).float()  # convert z,y,x -> x, y, z
        surface_verts = normalize_vertices(surface_verts, shape)

        perm = torch.randperm(surface_verts.shape[0])
        if len(perm)>point_count:
            perm = perm[: point_count]
        while(len(perm)<point_count):
            perm = torch.cat([perm,perm[:point_count - len(perm)]] ,dim= 0)
        # perm = perm[:min(len(perm),point_count)]
        surface_points_normalized_all = surface_verts[perm].cuda().unsqueeze(dim = 0)
        # 只取前3000个点
        # randomly pick 3000 points

    if mode == DataModes.TRAINING:
        return {'x': x,
                'y_voxels': y,
                'hot_map': hot_map,
                'surface_points': surface_points_normalized_all,
                'unpool': [0, 0, 0, 1, 1],
                # 'vertices_mc': verts,
                # 'faces_mc': faces,
                }
    else:
        return {'x': x,
                'y_voxels': y,
                'hot_map': hot_map,
                'surface_points': surface_points_normalized_all,
                'unpool': [0, 0, 0, 1, 1],
                # 'vertices_mc': verts,
                # 'faces_mc': faces,
                }

if __name__ == "__main__":
    # save_mesh()
    sys.path.append('/home/zhangzechu/文档/voxel2mesh-SOT')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from config import load_config
    from torch.utils.data import DataLoader
    cfg = load_config(0)
    cbct = CBCT()
    # cbct.pre_process_dataset(cfg)
    # cbct.save_data(cfg)
    datasets = cbct.quick_load_data(cfg)

    start = time.time()
    for i, datamode in enumerate([DataModes.TRAINING,DataModes.TESTING]):
        dataset = datasets[datamode]
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        for j,data in enumerate(loader):
            ct = data["x"]
            torch.cuda.empty_cache()
    end = time.time()
    print("time {}".format(end - start))