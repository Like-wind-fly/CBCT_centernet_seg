import os
import igl
import numpy as np
import torch
from neural_parts_code.PostProcess.visualize import get_merge_roi
from skimage import measure

def get_mesh(volume):
    reg_mask = volume.sum(dim=[0,1,2])>8
    if reg_mask == False:
        return False,None

    vertices_mc, faces_mc, _, _ = measure.marching_cubes_lewiner(volume.cpu().data.numpy(), 0.5 , step_size=1, allow_degenerate=False)
    vertices_mc = torch.from_numpy(vertices_mc).cuda()
    faces_mc = torch.from_numpy(faces_mc).long().cuda()
    return True,[vertices_mc,faces_mc]

def get_meshes(predictions, targets):
    pred_seg = predictions["volume"].int()
    gt_seg = targets["seg"].int()

    B, _, W, H, D = gt_seg.shape
    C = 32

    mask = torch.zeros([B,C]).bool().cuda()
    classify = torch.range(1,C).cuda()
    x_gt_meshes = []
    x_pred_meshes = []
    x_classify = []

    for b in range(B):
        gt_meshes_vert = []
        pred_meshes_vert = []
        gt_meshes_face = []
        pred_meshes_face = []

        for c in range(C):
            pred_volume = pred_seg[b,0] == (c+1)
            gt_volume = gt_seg[b,0] == (c+1)

            has_mesh1, gt_mesh = get_mesh(gt_volume)
            has_mesh2, pred_mesh = get_mesh(pred_volume)
            mask[b, c] = (has_mesh1 * has_mesh2)
            if has_mesh1 and has_mesh2:
                gt_meshes_vert.append(gt_mesh[0])
                gt_meshes_face.append(gt_mesh[1])
                pred_meshes_vert.append(pred_mesh[0])
                pred_meshes_face.append(pred_mesh[1])

        x_classify.append(classify[mask[b]])
        x_gt_meshes.append([gt_meshes_vert,gt_meshes_face])
        x_pred_meshes.append([pred_meshes_vert,pred_meshes_face])

    return x_classify,x_gt_meshes,x_pred_meshes





def get_volume(predictions,X,targets):
    # seg = targets["seg"]
    seg = X["quad_ct"]
    print(seg.shape)
    pred_seg = torch.zeros(seg.shape).cuda()

    B,C,W,H,D = seg.shape #对齐label大小

    volume_size = [W,H,D]

    # center = targets['center']  #对齐label volume中心
    center = predictions['center'].unsqueeze(0).cuda()
    quad_id = predictions['quad_id']
    # quad_id = targets['quad_id'] #对齐label volume 牙号

    pred_identify = predictions["quad_identify"] #预测八分区里的牙号
    # gt_identify   = targets["identify_seg"] #label牙号
    gt_identify   = predictions["identify_seg"]["quad_identify"] #label牙号
    B, C, W, H, D = gt_identify.shape
    B = 1
    C = 4
    pred_identify = pred_identify.view(1,4,8,W,H,D)

    print(quad_id)
    for i in range(B):
        _volume = torch.zeros(volume_size).cuda()[None].repeat(32,1,1,1)

        _pred_identify = pred_identify[i]
        _quad_id       = quad_id[i]
        print(_quad_id)
        _center        = center[i][_quad_id,:]
        _center = _center[:,None].repeat(1,8,1)
        _quad_id = torch.range(1,8)[None].repeat(C,1).cuda() + _quad_id[:,None].repeat(1,8)*8
        # _quad_id = torch.arange(1, 9)[None].repeat(C, 1).cuda() + _quad_id[:, None].repeat(1, 8) * 8

        _pred_identify = _pred_identify.view(C*8, W,H,D)
        _center        = _center.view(C * 8, 3)
        _quad_id       = _quad_id.view(C * 8).long()

        volume_i = get_merge_roi(rois=_pred_identify,center= _center, output_size= volume_size)
        blackground = (torch.sum(volume_i > 0.5,dim= [0]) == 0)[None].float() * 0.5
        _volume = torch.cat([blackground,_volume],dim=0)
        _volume[_quad_id] = volume_i
        _volume = torch.argmax(_volume,dim=0)

        pred_seg[i,0] = _volume

    return pred_seg

def get_volume2(predictions,targets):
    seg = targets["seg"]
    pred_seg = torch.zeros(seg.shape).cuda()
    B,C,W,H,D = seg.shape

    volume_size = [W,H,D]
    center = targets['center']
    quad_id = predictions['quad_seg']
    # quad_id = targets['quad_id']
    pred_identify = targets["identify_ct"]
    # pred_identify = predictions["quad_identify"]
    gt_identify   = targets["identify_seg"]

    B, C, W, H, D = gt_identify.shape
    # pred_identify = pred_identify.view(B,C,8,W,H,D)
    print(pred_identify.shape)

    for i in range(B):
        _volume = torch.zeros(volume_size).cuda()[None].repeat(32,1,1,1)

        _pred_identify = pred_identify[i]
        _quad_id       = quad_id[i]
        _center        = center[i][_quad_id,:]
        _center = _center[:,None].repeat(1,8,1)
        _quad_id = torch.range(1,8)[None].repeat(C,1).cuda() + _quad_id[:,None].repeat(1,8)*8

        _pred_identify = _pred_identify.view(C*8, W,H,D)
        _center        = _center.view(C * 8, 3)
        _quad_id       = _quad_id.view(C * 8).long()

        volume_i = get_merge_roi(rois=_pred_identify,center= _center, output_size= volume_size)
        blackground = (torch.sum(volume_i > 0.5,dim= [0]) == 0)[None].float() * 0.5
        _volume = torch.cat([blackground,_volume],dim=0)
        _volume[_quad_id] = volume_i
        _volume = torch.argmax(_volume,dim=0)

        pred_seg[i,0] = _volume

    return pred_seg








