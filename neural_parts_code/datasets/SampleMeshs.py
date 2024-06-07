import os
from os.path import join

import numpy as np
import torch
import pickle
from torch.utils.data import Dataset as dataset
from scipy.spatial.transform import Rotation as R
from neural_parts_code.datasets.transform import  stns
from skimage import measure
import  igl

class DataModes:
    TRAINING = 'training'
    VALIDATION = 'validation'
    TESTING = 'testing'
    ALL = 'all'
    def __init__(self):
        dataset_splits = [DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]

class SampleMesh():
    def __init__(self, image,hot_map, grad, surface_points , normals , cube_points , sign_dists , importantNum):
        self.image = image
        self.hot_map = hot_map
        self.grad = grad
        self.surface_points = surface_points
        self.normals = normals
        self.cube_points = cube_points
        self.sign_dists = sign_dists
        self.importantNum = importantNum
    # Points OUTSIDE the mesh will have NEGATIVE distance
    # Points within tol.merge of the surface will have POSITIVE distance
    # Points INSIDE the mesh will have POSITIVE distance

class SMDataset(dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self.readFromMen = False

    def __len__(self):
        # return 64
        return len(self.data)

    def __getitem__(self, idx):
        if self.readFromMen:
            item = self.data[idx]
        else:
            samplepath = self.data[idx]
            with open(samplepath, 'rb') as handle:
                item = pickle.load(handle)

        return get_item(item, self.mode)

    def readImageFromMen(self):
        dataset = []

        for idx in range(self.__len__()):
            samplepath = self.data[idx]
            with open(samplepath, 'rb') as handle:
                item = pickle.load(handle)
            dataset.append(item)
            print(idx)

        self.data = dataset
        self.readFromMen = True

def normalize_vertices(vertices, shape):

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
    vertices_mc = 2*(vertices_mc/(torch.max(shape)-1) - 0.5)

    return vertices_mc.cpu().numpy(), faces_mc

def get_item(item, mode):
    augmentation_shift_range = 5

    surface_points_num = item.surface_points.shape[0]
    cube_points_num = item.cube_points.shape[0]
    image = torch.tensor(item.image).view(1,64,64,64).cuda().float()
    hot_map = torch.tensor(item.hot_map).view(1,64,64,64).cuda().float()
    grad = torch.tensor(item.grad).view(1,64,64,64).cuda().float()
    sign_dists = torch.tensor(item.sign_dists).view(cube_points_num,1,1).cuda().float()
    cube_points = torch.tensor(item.cube_points).view(cube_points_num, 3).cuda().float()
    surface_points = torch.tensor(item.surface_points).view(surface_points_num,3).cuda().float()
    normals = torch.tensor(item.normals).cuda().view(surface_points_num,3).cuda().float()
    importantNum = item.importantNum


    hot_map[hot_map<0.995] = 0
    img = torch.cat([image, hot_map, grad], dim=0)

    # augmentation done only during training
    if mode == DataModes.TRAINING:  # if training do augmentation
        # 旋转
        theta_rotate = torch.eye(4).cuda()
        a = R.random().as_matrix()
        for i in range(3):
            for j in range(3):
                theta_rotate[i, j] = a[i, j]

        shift = torch.tensor(
            [d / (D // 2) for d, D in zip(2 * (torch.rand(3) - 0.5) * augmentation_shift_range,[64,64,64])])
        theta_shift = stns.shift(shift).cuda()

        f = 0.1
        scale = 1.0 - 2 * f * (torch.rand(1) - 0.5)
        theta_scale = stns.scale(scale).cuda()

        theta = theta_rotate @ theta_shift @ theta_scale

        img = stns.transform(theta, img)
        normals = stns.transformVert(theta_rotate, normals)
        surface_points = stns.transformVert(theta, surface_points)
        cube_points = stns.transformVert(theta, cube_points)

    normals = normals.view(surface_points_num, 1, 3)
    surface_points = surface_points.view(surface_points_num,1,3)
    cube_points = cube_points.view(cube_points_num, 1, 3)

    surface_normals = torch.cat([surface_points, normals], dim=-1)
    weight = sign_dists.clone()
    weight[weight<0] = 0.5
    weight[weight>0] = 1
    if importantNum>0:
        weight[:importantNum] = 2

    ocurr = sign_dists.clone()
    ocurr[ocurr<0] = 0
    ocurr[ocurr>0] = 1

    randOrderSurface = torch.randperm(surface_points_num)
    randOrderCube = torch.randperm(cube_points_num)

    surface_normals = surface_normals[randOrderSurface]
    cube_points = cube_points[randOrderCube]
    ocurr = ocurr[randOrderCube]
    weight = weight[randOrderCube]

    return [img[:2],cube_points,ocurr,weight,surface_normals,img[2:]]


class BuildSMDataSet():
    def __init__(self,data_root):
        self.data_root = data_root

        pass

    def quick_load_data(self):
        data_root = self.data_root
        down_data = {}

        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING, DataModes.VALIDATION]):
            dirPath = join(data_root, datamode)
            file_list = os.listdir(dirPath)
            file_list = ["{}/{}".format(dirPath,filename) for filename in file_list  if 'pickle' in filename]
            down_data[datamode] = SMDataset(file_list, datamode)

        return down_data

import meshplot as mp
if __name__ == "__main__":
    # save_mesh()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    buildDataSET = BuildSMDataSet("/home/zhangzechu/workspace/data/CBCT/neural_parts_root")
    datasets = buildDataSET.quick_load_data()

    for i, datamode in enumerate([DataModes.TRAINING,DataModes.TESTING,DataModes.VALIDATION]):
        dataset = datasets[datamode]
        for j,data in enumerate(dataset):
            surface_points = data[-1][:,:,:3].squeeze(dim = 1)
            normals = data[-1][:,:,3:].squeeze(dim = 1)
            cube_points = data[1].squeeze(dim = 1)
            sign_dists = data[2].squeeze()
            image = data[0][0]
            map = data[0][1]
            shape = torch.tensor([64,64,64])
            verts,faces = voxel2mesh(map, 1, shape)
            igl.write_obj("matchcube.obj",verts,faces)
            igl.write_obj("smooth.obj",surface_points.cpu().numpy(),np.array([[0,1,2]]))
            
            break
        break



