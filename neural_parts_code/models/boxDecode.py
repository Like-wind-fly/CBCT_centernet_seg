import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool3d(
        heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _topk(scores, K=48):
    batch, cat, depth, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (depth * height * width)
    topk_zs = (topk_inds / (height * width)).int().float()
    topk_ys = (topk_inds % (height * width) / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind/K).int()

    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_zs = _gather_feat(topk_zs.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_zs , topk_ys, topk_xs


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 4, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(4))
    feat = _gather_feat(feat, ind)
    return feat

def box_decode(hmap,reg,dwh,K=48,step_size = 4, centerMode = False):
    batch,C,D,H,W = hmap.shape

    hmap = _nms(hmap)  # perform nms on heatmaps
    scores, inds, clses, zs , ys, xs = _topk(hmap, K=K)
    reg = _transpose_and_gather_feat(reg, inds)

    reg = reg.view(batch, K, 3)

    zs = zs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    xs = xs.view(batch, K, 1) + reg[:, :, 2:3]
    zs *= step_size
    ys *= step_size
    xs *= step_size

    dwh = _transpose_and_gather_feat(dwh, inds)


    #返回中心和长方体长宽高，更容易计算
    if centerMode:
        center = torch.cat([zs,ys,xs],dim=2)
        return [center,dwh,scores,clses]

    #根据我自己的习惯，把box理解为范围
    bboxes = torch.cat([zs - dwh[..., 0:1] / 2,
                        zs + dwh[..., 0:1] / 2,
                        ys - dwh[..., 1:2] / 2,
                        ys + dwh[..., 1:2] / 2,
                        xs - dwh[..., 2:3] / 2,
                        xs + dwh[..., 2:3] / 2,
                       ]
                       , dim=2)

    return [bboxes,scores,clses]

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


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from neural_parts_code.datasets import build_datasets
    from scripts.utils import load_config
    from neural_parts_code.datasets.cbct.common import save_img

    config = load_config("/home/zhangzechu/workspace/neral-parts-CBCT/config/centernet.yaml")
    train_dataset, _, test_dataset = build_datasets(config)

    loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for j, data in enumerate(loader):
        ct = data[0][0, 0]
        seg = data[1][0, 0]
        print(ct.shape)
        print(seg.shape)
        # heatmap_tensor = data[2][0, 0]
        # reg_mask = data[3][0]
        # index_int = data[4][0]
        # reg = data[5][0]
        # dwh = data[6][0]




