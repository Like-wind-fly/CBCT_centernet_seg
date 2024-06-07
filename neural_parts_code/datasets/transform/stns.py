import torch.nn.functional as F
import torch
from neural_parts_code.datasets.transform import  affine_3d_grid_generator
from IPython import embed
import time
 

def stn_all_ratations(params, inverse=False):
    theta, theta_x, theta_y, theta_z = stn_all_ratations_with_all_theta(params, inverse)
    return theta

def stn_quaternion_rotations(params):

    params = params.view(3)
    qi, qj, qk = params

    s = qi ** 2 + qj ** 2 + qk ** 2

    theta = torch.eye(4, device=params.device)


    theta[0, 0] = 1 - 2 * s * (qj ** 2 + qk ** 2)
    theta[1, 1] = 1 - 2 * s * (qi ** 2 + qk ** 2)
    theta[2, 2] = 1 - 2 * s * (qi ** 2 + qj ** 2)

    theta[0, 1] = 2 * s * qi * qj
    theta[0, 2] = 2 * s * qi * qk

    theta[1, 0] = 2 * s * qi * qj
    theta[1, 2] = 2 * s * qj * qk

    theta[2, 0] = 2 * s * qi * qk
    theta[2, 1] = 2 * s * qj * qk

    return theta

 

def stn_batch_quaternion_rotations(params, inverse=False):
    thetas = []
    for param in params:
        theta = stn_quaternion_rotations(param)
        # if inverse:
        #     theta = theta.inverse()
        thetas.append(theta)

    thetas = torch.cat(thetas, dim=0)
    thetas = thetas.view(-1,4,4)
    return thetas

def scale(param):
    theta_scale = torch.eye(4) 

    theta_scale[0, 0] = param
    theta_scale[1, 1] = param
    theta_scale[2, 2] = param

    return theta_scale

def rotate(angles):
    angle_x, angle_y, angle_z = angles
    params = torch.Tensor([torch.cos(angle_x), torch.sin(angle_x), torch.cos(angle_y), torch.sin(angle_y),torch.cos(angle_z), torch.sin(angle_z)])
    params = params.view(3,2)
    theta = stn_all_ratations(params)

    return theta

 
def shift(axes):
    theta = torch.eye(4, device=axes.device)
    theta[0, 3] = axes[0]
    theta[1, 3] = axes[1]
    theta[2, 3] = axes[2]

    return theta

def transformVert(theta,verts):
    theta = theta.t()
    theta = theta.inverse()
    theta = theta[:,:3]
    new_verts = torch.ones((verts.shape[0],verts.shape[1]+1))
    if 'cuda' in verts.device.type:
        new_verts = new_verts.cuda()

    new_verts[:,:verts.shape[1]] = verts
    new_verts = new_verts @ theta

    return new_verts

def transform(theta, x , sample_mode = 'bilinear'):
    theta = theta[0:3, :].view(-1, 3, 4)

    grid = affine_3d_grid_generator.affine_grid(theta, x[None].shape)

    if 'cuda' in x.device.type:
        grid = grid.cuda()

    batchSize = x.shape[0]
    for i in range(batchSize):
        x[i] = F.grid_sample(x[i][None,None], grid, mode= sample_mode , padding_mode='zeros', align_corners=True)

    return x


