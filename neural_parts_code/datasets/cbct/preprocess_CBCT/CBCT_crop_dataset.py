import random
import pickle
import torch
import os

def splitDataset(dataset_path):
    dataset = {}
    for mode in [DataModes.TRAINING,DataModes.VALIDATION,DataModes.TESTING]:
        path = os.path.join(dataset_path,mode)
        file_list = os.listdir(path)
        file_list = [os.path.join(path,fname) for fname in file_list if "pickle" in fname]
        dataset[mode] = file_list

    with open(os.path.join(dataset_path,"splitGroup.pt"), 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

dataset_path = "/home/zhangzechu/workspace/data/CBCT150/CBCT_CROP_122/"
splitDataset(dataset_path)

from torch.nn import functional as F
from torchvision.ops import masks_to_boxes

from torch.utils.data import Dataset as dataset
from torch.utils.data import DataLoader
from common import centerCrop3d,GaussianSmoothing,BasicSample,save_img
from config import args
from skimage import measure
import io
import igl
import numpy as np
import math



#tensor_list 里的 D,W,H = tensor.shape
def roiCrop(tensor_list,center,sizes):
    tensor_shape = tensor_list[0].shape

    crop = []
    for i in range(3):
        crop.append([center[i] - math.floor(sizes[i] / 2.0),center[i] + math.ceil(sizes[i]  / 2.0)])
    crop = torch.tensor(crop).long()
    crop_over = crop.clone()
    crop_over[crop_over<0] = 0

    tensor_list = [x[crop_over[0][0]:crop_over[0][1], crop_over[1][0]:crop_over[1][1], crop_over[2][0]:crop_over[2][1]].clone() for x in tensor_list]

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
    pad = torch.nn.ConstantPad3d(padding, value=0)
    tensor_list = [pad(x[None,None])[0,0] for x in tensor_list]

    return tensor_list

def cropInput(X, centers, dwhs, scores,normal_size = 64,offset = 8):
    B, C, _ = centers.shape

    mask = scores > 0.5
    rois = [[None for j in range(C)] for i in range(B)]
    cropsize = torch.zeros(mask.shape).cuda()
    for i in range(B):
        for boxId in range(len(centers[i])):
            exist_ = scores[i][boxId] > 0.5

            if exist_:
                boxSize = max(normal_size, torch.max(dwhs[i][boxId]) + offset)
                boxSize += boxSize % 2
                cropsize[i][boxId] = boxSize
                center = centers[i][boxId]

                roi = roiCrop([X[i, 0]], center, [boxSize, boxSize, boxSize])[0]
                rois[i][boxId] = roi


    return rois, mask, cropsize

class DataModes:
    TRAINING = 'training'
    VALIDATION = 'validation'
    TESTING = 'testing'
    ALL = 'all'
    def __init__(self):
        dataset_splits = [DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]

#这个函数是为了减少每个batch处理的牙齿个数，从而可以设置更高的Batch size
# limit_teeth_num 是限制返回的牙齿对数， 返回的牙齿数应该是 2* limit_teeth_num
# limit_teeth_num上限
# 口腔中牙齿序号为
# 1 2
# 0 3
def sampleFromTeeth(limit_teeth_num):

    if limit_teeth_num>8:
        x_random = [0,2]
        limit_teeth_num= int(limit_teeth_num/2)

    elif limit_teeth_num<=8:
        x_random = [random.randint(0, 1) * 2]


    mask = torch.zeros(32)
    y_random = random.randint(0, 8 - limit_teeth_num)
    for i in range(limit_teeth_num):
        for j in x_random:
            mask[j*8 + y_random + i] = 1
            mask[j*8+ 8 + y_random + i] = 1

    return mask>0

def smooth_verts(verts, faces,step = 3):
    for i in range(step):
        verts = igl.per_vertex_attribute_smoothing(verts, faces)
    return verts

def get_curvature_weight(verts, faces):
    v1, v2, k1, k2 = igl.principal_curvature(verts, faces)
    curve = np.abs(k1)+np.abs(k2)
    thead = np.mean(curve)
    curve[curve<thead] = thead
    curve[curve>thead*2] = thead*2
    curve = (curve - thead)/thead
    return curve

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class CBCTDataset(dataset):
    def __init__(self, data, mode, configs):
        self.data = data
        self.mode = mode
        self.configs = configs
        exec('self.limit_teeth_num = configs.{}_limit_teeth_num'.format(mode))
        self.readFromMen = False

    def __len__(self):
        # return min(len(self.data),10)
        return len(self.data)

    def readImageFromMen(self):
        dataset = []

        for idx in range(self.__len__()):
            samplepath = self.data[idx]
            with open(samplepath, 'rb') as handle:
                # roi_ct, roi_seg, cropsize, reg_mask, index_int, dwh , all_suface_point,all_point_weight, roi_near_tooth = pickle.load(handle)
                roi_ct, roi_seg, cropsize, reg_mask, index_int, dwh, all_suface_point, all_point_weight, roi_near_tooth = CPU_Unpickler(handle).load()

            dataset.append([roi_ct, roi_seg, cropsize, reg_mask, index_int, dwh , all_suface_point,all_point_weight , roi_near_tooth])
            print(idx)

        self.data = dataset
        self.readFromMen = True

    def __getitem__(self, idx):
        samplepath = self.data[idx]
        with open(samplepath, 'rb') as handle:
            ct, seg = pickle.load(handle)

        id_for_cbct = self.data[idx]
        id_for_cbct = int(id_for_cbct[56:-7])
        if id_for_cbct>99:
            return id_for_cbct

        with torch.no_grad():
            ct = ct.float().cuda()
            seg = seg.float().cuda()

            ct, seg = augmentation(ct,seg, self.configs)
            seg,reg_mask,index_int,dwh = process(seg)

            roi_seg,_,cropsize = cropInput(seg[None,None], index_int[None], dwh[None], reg_mask[None], normal_size = 64, offset = 2)
            roi_ct,_,_  = cropInput(ct[None,None], index_int[None], dwh[None], reg_mask[None], normal_size=64, offset=2)
            roi_ct = roi_ct[0]
            roi_seg = roi_seg[0]

            roi_near_tooth = []
            for i in range(len(roi_seg)):
                if reg_mask[i]:
                    roi_near_tooth.append(roi_seg[i].clone())
                else:
                    roi_near_tooth.append(None)

            for i in range(len(roi_seg)):
                if reg_mask[i]:
                    roi_seg[i] = (roi_seg[i]==(i+1))
                    near_tooth = roi_near_tooth[i]
                    near_tooth[roi_seg[i]] = 0
                    near_tooth[near_tooth>0] = 1
                    roi_near_tooth[i] = near_tooth

            # pad = torch.nn.ConstantPad3d(padding=3 // 2, value=0).cuda()
            # guass = GaussianSmoothing(1, 3, 0.7, dim=3).cuda()
            # for i,roi in enumerate(roi_seg):
            #     if roi != None:
            #         roi = roi.cuda()
            #         seg = guass(pad(roi.float()[None, None]))[0, 0]
            #         shape = torch.tensor([seg.shape[0],seg.shape[1],seg.shape[2]])
            #         surface_points,faces = voxel2mesh(seg,1,shape)
            #         v = surface_points.cpu().numpy()
            #         f = faces.cpu().numpy()
            #         igl.write_obj("/home/zhangzechu/workspace/data/CBCT150/CBCT_CROP_122/teethOBJ/tooth_{}_{}_{}.obj".format(self.mode,idx,i),v,f)
            # print("tooth_{}_{}".format(self.mode,idx))
            # return "tooth_{}_{}".format(self.mode,idx)

            point_count = 3000
            all_suface_point = torch.zeros(32, point_count, 3).cuda()
            all_point_Comform = torch.zeros(32, point_count, 3).cuda()

            for i,roi in enumerate(roi_seg):
                if roi != None:
                    roi = roi.cuda()
                    toothName = "tooth_{}_{}_{}.obj".format(self.mode,idx,i)
                    sourcemesh = os.path.join("/home/zhangzechu/workspace/data/CBCT150/CBCT_CROP_122/teethRemeshOBJ", toothName)
                    targetmesh = os.path.join("/home/zhangzechu/workspace/data/CBCT150/CBCT_CROP_122/teethComformOBJ", toothName)
                    V_S, _, n, F, _, _ = igl.read_obj(sourcemesh)
                    V_T, _, n, F, _, _ = igl.read_obj(targetmesh)
                    if len(V_T)  != len(V_T):
                        print("bugs vertices")

                    perm = torch.randperm(len(V_T))
                    while len(perm) < point_count:
                        perm = torch.cat([perm, perm[:point_count - len(perm)]],dim=0)

                    V_S = torch.tensor(V_S)
                    V_T = torch.tensor(V_T)
                    V_S = V_S[perm[:min([len(perm), point_count])]].cuda()
                    V_T = V_T[perm[:min([len(perm), point_count])]].cuda()

                    all_suface_point[i] =  V_S
                    all_point_Comform[i] = V_T

            for i,roi in enumerate(roi_seg):
                if roi != None:
                    roi_ct[i] = roi_ct[i].cpu()
                    roi_seg[i] = roi_seg[i].cpu()
                    roi_near_tooth[i] = roi_near_tooth[i].cpu()
            if not os.path.isdir("/home/zhangzechu/workspace/data/CBCT150/CBCT_CROP_97/{}".format(self.mode)):
                os.mkdir("/home/zhangzechu/workspace/data/CBCT150/CBCT_CROP_97/{}".format(self.mode))

            with open("/home/zhangzechu/workspace/data/CBCT150/CBCT_CROP_97/{}/{}.pickle".format(self.mode,idx),'wb') as handle:
                pickle.dump([roi_ct,roi_seg,cropsize,reg_mask.cpu(),index_int.cpu(),dwh.cpu(),all_suface_point.cpu(),all_point_Comform.cpu(),roi_near_tooth], handle, protocol=pickle.HIGHEST_PROTOCOL)
            return cropsize





def normalize_vertices(vertices, shape):
    assert len(vertices.shape) == 2, "Inputs must be 2 dim"

    return 2*(vertices/(torch.max(shape)-1) - 0.5)

def voxel2mesh(volume, gap, shape):
    '''
    :param volume:
    :param gap:
    :param shape:
    :return:
    '''
    vertices_mc, faces_mc, _, _ = measure.marching_cubes_lewiner(volume.cpu().data.numpy(), 0.5 , step_size=gap, allow_degenerate=False)
    vertices_mc = torch.flip(torch.from_numpy(vertices_mc), dims=[1]).float()  # convert z,y,x -> x, y, z
    vertices_mc = normalize_vertices(vertices_mc, shape)
    faces_mc = torch.from_numpy(faces_mc).long()
    return vertices_mc, faces_mc




def getTransformMax(index_int,cropsize):
    Batch = index_int.shape[0]
    mat = torch.zeros(Batch,4,4).cuda()
    for i in range(Batch):
        factor = (cropsize[i] - 1)/2.0
        mat[i, 0, 2] = factor
        mat[i, 1, 1] = factor
        mat[i, 2, 0] = factor
        mat[i, 3, 3] = 1
        mat[i,:3,3] = index_int[i,:] - 0.5 + (cropsize[i]%2)/2.0
    return mat


def process(seg):
    # 将八进制的mask标注转为十进制
    D, W, H = seg.shape
    one_map = torch.stack([ seg == i for i in range(1,33)],dim=0)


    reg_mask = one_map.sum(dim=[1,2,3])>25
    existTeeth = one_map[reg_mask]

    value = torch.arange(1, 33).cuda()

    seg = torch.zeros(seg.shape).cuda()
    for i, v in enumerate(value):
        seg = torch.stack([seg, v *one_map[i]],dim=0).sum(dim=0)

    #根据mask求包围框
    ETMaskDW = existTeeth.sum(dim = -1)>0
    ETMaskDH = existTeeth.sum(dim=  2)>0

    rangeDW = masks_to_boxes(ETMaskDW)
    rangeDH = masks_to_boxes(ETMaskDH)
    rangeDWH = torch.cat([rangeDW[:,[1,3]],rangeDW[:,[0,2]],rangeDH[:,[0,2]]],dim= 1)
    dwh = torch.stack([(rangeDWH[:,i * 2 + 1] - rangeDWH[:,i * 2]) for i in range(3)], dim=1)
    index = torch.stack([(rangeDWH[:,i * 2 + 1] + rangeDWH[:,i * 2]) for i in range(3)], dim=1)
    index = (index/2).int()
    index_int = index.int()

    D,W,H = seg.shape


    temp = torch.zeros(32,3).cuda().int()
    temp[reg_mask] = index_int
    index_int = temp

    temp = torch.zeros(32,3).cuda().int()
    temp[reg_mask] = dwh.int()
    dwh = temp

    return seg,reg_mask,index_int,dwh


def augmentation(ct_tensor,seg_tensor, configs):
    # augmentation done only during training
    test_crop_size = configs.test_crop_size

    ct_tensor = centerCrop3d(ct_tensor[None], test_crop_size)
    seg_tensor = centerCrop3d(seg_tensor[None], test_crop_size)

    return ct_tensor[0], seg_tensor[0]

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
        args.test_crop_size = conf["data"].get("test_crop_size",args.test_crop_size)
        args.train_crop_size = conf["data"].get("train_crop_size", args.train_crop_size)
        args.training_limit_teeth_num = conf["training"].get("limit_teeth_num", 32)
        args.validation_limit_teeth_num = conf["validation"].get("limit_teeth_num", 32)
        args.testing_limit_teeth_num = conf["testing"].get("limit_teeth_num", 32)
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





def getVerticeMask(vertices, mask):
    # vertices.shape = B,N,3
    B, C , D, H, W = mask.shape
    factor = W - 1
    vertices = factor * (vertices + 1) / 2
    vertices = torch.round(vertices).long()

    index = H * W * vertices[:, :, 2] + W * vertices[:, :, 1] + vertices[:, :, 0]
    for i in range(index.shape[0]):
        index[i] += D * H * W * i

    index = index.view(-1)
    mask = mask.view(-1)
    mask[index] = 1
    return mask.view(B, 1, D, H, W)


import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config

if __name__ == "__main__":
    pass
    # print(sampleFromTeeth(12))


    # # from neural_parts_code.PostProcess.TF_visualize import all_teeth_Meshes
    # # config = load_config("/home/zhangzechu/workspace/neral-parts-CBCT/config/teethflow_dataset.yaml")
    # config = load_config("/home/zhangzechu/workspace/neral-parts-CBCT/config/teethflow_TOWN10_sample122.yaml")
    # datasets = BuildCBCTDataSet(config).quick_load_data()
    # # train_dataset = datasets["training"]
    # # Val_dataset = datasets["validation"]
    # # test_dataset = datasets["testing"]
    #
    # torch.cuda.set_device(int(config["network"]["cuda"]))
    # sampler = BasicSample()
    # save_path = "/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/gt"
    #
    # # 定义数据加载
    # for i, dataset in enumerate(datasets.values()):
    #     loader = DataLoader(dataset, batch_size=1, shuffle=False)
    #     for j,data in enumerate(loader):
    #         pass
    # #         # roi_seg = data["roi_seg"][0]
    # #         # reg_mask = data["reg_mask"][0]
    # #         # all_suface_point = data["all_suface_point"][0]
    # #         # all_point_Comform = data["all_point_Comform"][0]
    # #         #
    # #         # y_prim = []
    # #         # phi_faces = []
    # #         # comform_prim = []
    # #         #
    # #         # for k in range(32):
    # #         #     if reg_mask[k]:
    # #         #         # seg = roi_seg[k]
    # #         #         # shape = torch.tensor([seg.shape[0],seg.shape[1],seg.shape[2]])
    # #         #         # seg>=0.5
    # #         #         # matching_cube_point,faces = voxel2mesh(seg,1,shape)
    # #         #
    # #         #         toothName = "tooth_testing_{}_{}.obj".format(j,k)
    # #         #         sourcemesh = os.path.join("/home/zhangzechu/workspace/neral-parts-CBCT/output/teethRemeshOBJ", toothName)
    # #         #         targetmesh = os.path.join("/home/zhangzechu/workspace/neral-parts-CBCT/output/teethComformOBJ", toothName)
    # #         #         V_S, _, n, F, _, _ = igl.read_obj(sourcemesh)
    # #         #         V_T, _, n, F, _, _ = igl.read_obj(targetmesh)
    # #         #
    # #         #         y_prim.append(torch.tensor(V_S))
    # #         #         phi_faces.append(torch.tensor(F))
    # #         #         comform_prim.append(torch.tensor(V_T))
    # #         #
    # #         #
    # #         # predictions = {}
    # #         # predictions["y_prim"] = y_prim
    # #         # predictions["phi_faces"] = phi_faces
    # #         #
    # #         # # pred_verts,pred_faces,mesh_class = all_teeth_Meshes(predictions,data)
    # #         # #
    # #         # # torch.save([pred_verts, pred_faces, mesh_class], os.path.join(save_path, "all_teeth_{}.pt".format(j)))
    # #         # # torch.save([comform_prim, pred_faces, mesh_class],os.path.join(save_path, "all_teeth_comform_{}.pt".format(j)))










