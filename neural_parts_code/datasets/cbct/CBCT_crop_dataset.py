import random
import pickle
import torch
import os
from torch.nn import functional as F
from torchvision.ops import masks_to_boxes
from torch.utils.data import Dataset as dataset
from torch.utils.data import DataLoader
from neural_parts_code.datasets.cbct.common import centerCrop3d,GaussianSmoothing,BasicSample
from scipy.spatial.transform import Rotation as R
from neural_parts_code.datasets.transform import  stns,affine_3d_grid_generator
from neural_parts_code.datasets.cbct.config import args
from skimage import measure
import io
import igl
import numpy as np

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

    mask = mask > 0
    return mask

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
        if self.readFromMen:
            roi_ct, roi_seg, cropsize, reg_mask, index_int, dwh , all_suface_point,all_point_Comform , roi_near_tooth = self.data[idx]
        else:
            samplepath = self.data[idx]
            with open(samplepath, 'rb') as handle:
                # ct,seg = pickle.load(handle)
                # roi_ct, roi_seg, cropsize, reg_mask, index_int, dwh , all_suface_point,all_point_Comform , roi_near_tooth = pickle.load(handle)
                roi_ct, roi_seg, cropsize, reg_mask, index_int, dwh , all_suface_point,all_point_Comform , roi_near_tooth = CPU_Unpickler(handle).load()
                # roi_ct, roi_seg, cropsize ,reg_mask,index_int,dwh = pickle.load(handle)

        with torch.no_grad():
            mask = sampleFromTeeth(self.limit_teeth_num)

            num = reg_mask[mask].sum()
            while(num == 0):
                mask = sampleFromTeeth(self.limit_teeth_num)
                num = reg_mask[mask].sum()

            reg_mask = reg_mask[mask].cuda()

            roi_ct = [roi_ct[i] for i in range(32) if mask[i]]
            roi_seg = [roi_seg[i] for i in range(32) if mask[i]]
            roi_near_tooth = [roi_near_tooth[i] for i in range(32) if mask[i]]

            cropsize = cropsize[0]
            cropsize = cropsize[mask].cuda()

            surface_points = all_suface_point[mask].cuda()
            all_point_Comform = all_point_Comform[mask].cuda()
            index_int = index_int[mask].cuda()
            dwh = dwh[mask].cuda()

            for i,roi in enumerate(roi_seg):
                if roi != None:
                    roi_ct[i] = roi_ct[i].cuda()
                    roi_seg[i] = roi_seg[i].cuda()
                    roi_near_tooth[i] = roi_near_tooth[i].cuda()


            global_mat = getTransformMax(index_int,cropsize).cuda()
            local_mat,roi_ct,roi_seg,roi_near_tooth,all_suface_point = resize(roi_ct,roi_seg,roi_near_tooth,surface_points, reg_mask,self.mode,normal_size = 64)

            return {
                   'roi_ct': roi_ct,
                   'reg_mask': reg_mask,
                   'global_mat': global_mat,
                   'local_mat': local_mat,
                   'roi_seg':roi_seg,
                   'all_suface_point': all_suface_point,
                   'all_point_Comform':all_point_Comform,
                   'roi_near_tooth': roi_near_tooth
                   }


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

def resize(roi_ct,roi_seg,roi_near_tooth,points,reg_mask,mode,normal_size = 64):
    augmentation_shift_range = 2

    if mode == DataModes.TRAINING:  # if training do augmentation
        # 旋转
        theta_rotate = torch.eye(4)
        a = R.random().as_matrix()
        for i in range(3):
            for j in range(3):
                theta_rotate[i, j] = a[i, j]

        shift = torch.tensor(
            [d / (D // 2) for d, D in zip(2 * (torch.rand(3) - 0.5) * augmentation_shift_range, [normal_size, normal_size, normal_size])])
        theta_shift = stns.shift(shift)

        f = 0.1
        scale = 1.0 - 2 * f * (torch.rand(1) - 0.5)
        theta_scale = stns.scale(scale)

        theta = theta_rotate @ theta_shift @ theta_scale
    else:
        theta =  torch.eye(4).cuda()

    B = len(roi_ct)
    new_roi_ct = torch.zeros([B, normal_size, normal_size, normal_size]).cuda()
    new_roi_seg = torch.zeros([B, normal_size, normal_size, normal_size]).cuda()
    new_roi_near_tooth = torch.zeros([B, normal_size, normal_size, normal_size]).cuda()

    shape = new_roi_ct[0][None,None].shape
    grad_theta = theta[0:3, :].view(-1, 3, 4)
    grid = affine_3d_grid_generator.affine_grid(grad_theta.cuda(), shape)
    kernel_size = 3;sigma = 0.5
    pad = torch.nn.ConstantPad3d(padding=kernel_size // 2, value=0).cuda()
    guass = GaussianSmoothing(1, kernel_size, sigma, dim=3).cuda()
    point_count = 3000
    all_suface_point = torch.zeros(B,point_count,3).cuda()

    for i in range(B):
        if not reg_mask[i]:
            continue
        roi_seg[i] = guass(pad(roi_seg[i].float()[None,None]))[0,0]
        roi_near_tooth[i] = guass(pad(roi_near_tooth[i].float()[None, None]))[0, 0]

        if mode == DataModes.TRAINING or roi_seg[i].shape[0] != normal_size:
            new_roi_ct[i]  = F.grid_sample(roi_ct[i][None,None], grid, mode="bilinear", padding_mode='zeros', align_corners=True)
            new_roi_seg[i] = F.grid_sample(roi_seg[i][None,None], grid, mode="bilinear", padding_mode='zeros', align_corners=True)
            new_roi_near_tooth[i] = F.grid_sample(roi_near_tooth[i][None, None], grid, mode="bilinear", padding_mode='zeros',align_corners=True)
        else :
            new_roi_ct[i] = roi_ct[i]
            new_roi_seg[i] = roi_seg[i]
            new_roi_near_tooth[i] = roi_near_tooth[i]

        all_suface_point[i] = stns.transformVert(theta.cuda(), points[i])[None]

    if mode == DataModes.TRAINING:
        meanv = (random.random() - 0.5)/5
        sizeoff = 1+(random.random() - 0.5)/5
        new_roi_ct = (meanv+new_roi_ct) * sizeoff

    theta = theta[None].repeat(B,1,1)

    return theta,new_roi_ct,new_roi_seg,new_roi_near_tooth,all_suface_point

def verticesToIndex(global_mat,local_mat,vertices):
    return None

def IndexsToVertices(global_mat,local_mat,indexs):
    return None

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
    one_map_id = []
    for i in range(1, 5):
        for j in range(1, 9):
            one_map_id.append(i * 10 + j)
    one_map_id = torch.tensor(one_map_id).cuda().int()
    # 11,12,13  -> 9,10,11  八 -》 十

    one_map = torch.stack([ seg == one_map_id[i] for i in range(one_map_id.shape[0])],dim=0)


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


def splitDataset(dataset_path):
    dataset = {}
    for mode in [DataModes.TRAINING,DataModes.VALIDATION,DataModes.TESTING]:
        path = os.path.join(dataset_path,mode)
        file_list = os.listdir(path)
        file_list = [os.path.join(path,fname) for fname in file_list if "pickle" in fname]
        dataset[mode] = file_list

    with open(os.path.join(dataset_path,"splitGroup.pt"), 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


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


if __name__ == "__main__":
    from neural_parts_code.datasets import build_datasets
    from scripts.utils import load_config
    from neural_parts_code.PostProcess.TF_visualize import all_teeth_Meshes

    # config = load_config("/home/zhangzechu/workspace/neral-parts-CBCT/config/teethflow_dataset.yaml")
    config = load_config("/home/zhangzechu/workspace/neral-parts-CBCT/config/teethflow_TOWN_Comform.yaml")
    datasets = build_datasets(config)
    torch.cuda.set_device(int(config["network"]["cuda"]))
    sampler = BasicSample()
    save_path = "/home/zhangzechu/workspace/neral-parts-CBCT/demo/output/gt"

    # 定义数据加载
    for i, dataset in enumerate(datasets[2:]):
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for j,data in enumerate(loader):
            roi_seg = data["roi_seg"][0]
            reg_mask = data["reg_mask"][0]
            all_suface_point = data["all_suface_point"][0]
            all_point_Comform = data["all_point_Comform"][0]

            y_prim = []
            phi_faces = []
            comform_prim = []

            for k in range(32):
                if reg_mask[k]:
                    # seg = roi_seg[k]
                    # shape = torch.tensor([seg.shape[0],seg.shape[1],seg.shape[2]])
                    # seg>=0.5
                    # matching_cube_point,faces = voxel2mesh(seg,1,shape)

                    toothName = "tooth_testing_{}_{}.obj".format(j,k)
                    sourcemesh = os.path.join("/home/zhangzechu/workspace/neral-parts-CBCT/output/teethRemeshOBJ", toothName)
                    targetmesh = os.path.join("/home/zhangzechu/workspace/neral-parts-CBCT/output/teethComformOBJ", toothName)
                    V_S, _, n, F, _, _ = igl.read_obj(sourcemesh)
                    V_T, _, n, F, _, _ = igl.read_obj(targetmesh)

                    y_prim.append(torch.tensor(V_S))
                    phi_faces.append(torch.tensor(F))
                    comform_prim.append(torch.tensor(V_T))


            predictions = {}
            predictions["y_prim"] = y_prim
            predictions["phi_faces"] = phi_faces

            pred_verts,pred_faces,mesh_class = all_teeth_Meshes(predictions,data)

            torch.save([pred_verts, pred_faces, mesh_class], os.path.join(save_path, "all_teeth_{}.pt".format(j)))
            torch.save([comform_prim, pred_faces, mesh_class],os.path.join(save_path, "all_teeth_comform_{}.pt".format(j)))










