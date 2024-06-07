import os
import igl
import numpy as np
import torch

# output_size = [W,H,D]
# rois = [roi1,roi2,...,roin]
# center.shape = N,3
def get_merge_roi(rois,center,output_size):
    N,_ = center.shape

    roi_size = center.clone()
    volume = torch.zeros(output_size).cuda()
    volume = volume[None].repeat(N,1,1,1)

    for i in range(N):
        W,H,D = rois[i].shape
        roi_size[i][0] = W
        roi_size[i][1] = H
        roi_size[i][2] = D

    volume_range,roi_range = get_intersect(output_size,roi_size,center)

    for i in range(N):
        roi = rois[i]
        roi = roi[roi_range[i, 0]:roi_range[i, 1], roi_range[i, 2]:roi_range[i, 3],roi_range[i, 4]:roi_range[i, 5]]
        volume[i, volume_range[i, 0]:volume_range[i, 1], volume_range[i, 2]:volume_range[i, 3],volume_range[i, 4]:volume_range[i, 5]] = roi

    return volume

# volume_size = [W,H,D]
# roi_size.shape = N,3
# roi_center.shape = N,3
def get_intersect(volume_size,roi_size,roi_center):
    N,_ = roi_center.shape
    volume_size = torch.tensor(volume_size)[None].repeat(N,1).cuda()

    roi_low_bound  = roi_center - torch.floor(roi_size/2)
    roi_high_bound = roi_center + torch.ceil(roi_size/2)

    # 低于最低边界，超过值设为正数
    roi_over_low_bound = roi_low_bound.clone()
    roi_over_low_bound[roi_over_low_bound>0] = 0
    roi_over_low_bound*=(-1)

    # 高于最高边界，超过值设为负数
    roi_over_high_bound = volume_size - roi_high_bound
    roi_over_high_bound[roi_over_high_bound>0] = 0

    volume_AND_low = roi_low_bound + roi_over_low_bound
    volume_AND_high = roi_high_bound + roi_over_high_bound

    roi_AND_low =  roi_over_low_bound
    roi_AND_high = roi_size + roi_over_high_bound

    volume_range = torch.cat([volume_AND_low,volume_AND_high],dim=1)[:,[0,3,1,4,2,5]].int()
    roi_range = torch.cat([roi_AND_low, roi_AND_high], dim=1)[:, [0, 3, 1, 4, 2, 5]].int()

    return volume_range,roi_range
