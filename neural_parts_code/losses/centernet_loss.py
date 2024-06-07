import torch
import torch.nn as nn

from ..stats_logger import StatsLogger


def heatmap32_loss(predictions, targets, config):
    gt = targets["heatmap32"]
    pred = predictions["classify"]

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    StatsLogger.instance()["heatmap32_loss"].value = loss.item()
    return loss


def heatmap_loss(predictions, targets, config):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x d x h x w)
        gt_regr (batch x c x d x h x w)
    '''
    # gt = targets[1]

    gt = targets["heatmap"]
    pred = predictions["heatmap"]

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    StatsLogger.instance()["heatmap_loss"].value = loss.item()
    return loss


def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def sample_feature(feature,mask,ind):
    # feature :  batch C D H W
    # mask    :  batch 32
    # index   :  batch 32 3
    # samples :  batch 32 C
    B,C,D,H,W = feature.shape
    B,N = mask.shape
    index = ind.clone()

    index[:, :, 0] *= (H * W)
    index[:, :, 1] *= (W)
    index = index.sum(dim = -1)
    batch_index = torch.range(0,B-1).cuda()[:,None].repeat(1,N) * D * H * W
    index =batch_index + index


    mask = mask.view(-1)
    index = index.view(-1)
    samples = torch.zeros([B * N, C]).cuda()

    exist_obj = index[mask].long()

    feature = feature.permute(0, 2, 3, 4, 1)
    feature = feature.reshape(-1,C)
    out = feature[exist_obj,:]

    samples[mask] = out

    return samples.view(B,N,C)

def offset_loss(predictions, targets, config):
    # gt_mask =  targets[2]
    # gt_index = targets[3]
    #
    # gt_reg =   targets[4]

    gt_mask =  targets["reg_mask"]
    gt_index = targets["index_int"]
    gt_reg =   targets["reg"]

    pred_reg = predictions["reg"]

    sample_reg = sample_feature(pred_reg,gt_mask,gt_index)

    loss = _reg_loss(sample_reg, gt_reg, gt_mask)
    StatsLogger.instance()["offset_loss"].value = loss.item()
    return loss

def size_loss(predictions, targets, config):
    gt_mask =  targets["reg_mask"]
    gt_index = targets["index_int"]
    gt_dwh =   targets["dwh"]

    pred_dwh = predictions["dwh"]
    # test_sample_feature(pred_dwh,gt_mask,gt_index)

    sample_dwh = sample_feature(pred_dwh,gt_mask,gt_index)
    loss = _reg_loss(sample_dwh, gt_dwh, gt_mask)
    StatsLogger.instance()["size_loss"].value = loss.item()
    return loss

def class_ce_loss(predictions, targets, config):
    gt_mask =  targets["reg_mask"]
    gt_index = targets["index_int"]
    B, C = gt_mask.shape

    pred_class = predictions["classify"]
    classNum = pred_class.shape[1]

    gt_class = torch.arange(0,32).cuda()
    gt_class %= classNum
    gt_class = gt_class[None].repeat(B,1)

    sample_class = sample_feature(pred_class,gt_mask,gt_index)

    loss = torch.tensor(0).float().cuda()
    correct_avg = torch.tensor(0).float().cuda()

    if gt_mask.sum() > 0:
        gt_mask = gt_mask.view(-1)
        sample_class = sample_class.view(B * C, -1)[gt_mask]
        gt_class = gt_class.view(B * C)[gt_mask]
        ce_loss = torch.nn.CrossEntropyLoss()

        # labels = torch.FloatTensor([[0, 1], [1, 0], [1, 0])
        one_hot = torch.nn.functional.one_hot(gt_class,32).cuda().float()
        loss = torch.nn.BCELoss()(sample_class,one_hot)
        # loss = ce_loss(sample_class, gt_class)

        sample_class = torch.argmax(sample_class,dim=1)
        correct = sample_class == gt_class
        correct_avg = correct.sum()/gt_mask.sum()

    StatsLogger.instance()["ce_loss"].value = loss.item()
    StatsLogger.instance()["correct_classify"].value = correct_avg.item()
    return loss

def test_sample_feature(feature,mask,ind):
    test_feature = feature.clone()
    test_feature *= 0

    B,N = mask.shape

    count = 1
    for i in range(B):
        for j in range(N):
            if mask[i,j] :
                d,h,w = ind[i,j]
                test_feature[i,:,d,h,w] = count
                count+=1
    dd = test_feature.max()
    out = sample_feature(test_feature,mask,ind)
    x = out[:,0]
    x = x[mask]
    pass




