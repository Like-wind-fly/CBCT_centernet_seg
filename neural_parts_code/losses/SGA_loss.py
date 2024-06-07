import torch
import torch.nn as nn


from ..stats_logger import StatsLogger

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred = pred.squeeze(dim=1)

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].sum(dim=1).sum(dim=1).sum(dim=1) +
                                                target[:,i].sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        # 返回的是dice距离
        dice = dice / pred.size(1)
        loss = (1 - dice)
        meanloss = loss.mean()
        return meanloss

def dice_loss(pred, target):
    Union = pred * target
    AND = pred + target
    smooth = 1
    loss = 1 - 2*Union.sum(dim = [1,2,3,4])/(AND.sum(dim = [1,2,3,4])+smooth)
    loss = loss.mean()
    return loss


def tm_SGA_loss(predictions, targets, config):
    lossFunction = dice_loss
    ce_lossFunction = nn.BCELoss()
    loss = torch.tensor(0).float().cuda()

    gt_seg = targets["quad_seg"]

    B,C,W,H,D = gt_seg.shape
    gt_seg = gt_seg.view(B*C,W,H,D).unsqueeze(dim=1)

    gt_identify = targets["identify_seg"]
    B, C, W, H, D = gt_identify.shape
    gt_identify = gt_identify.view(B * C, W, H, D).unsqueeze(dim=1)

    gt_identify = torch.cat([gt_identify == i for i in range(1, 9)], dim=1)
    down_sample = nn.AvgPool3d(kernel_size = (16,16,16), stride = (16,16,16))



    gt_identify_down = down_sample(gt_identify.float())
    gt_identify_down = (gt_identify_down > 0.3).float()

    pred_identify = predictions["quad_identify"]
    pred_identify_down = predictions["down_sample_classify"]
    # pred_identify = nn.functional.sigmoid(pred_identify)

    loss_identify = lossFunction(gt_identify, pred_identify)
    loss_identify_down = lossFunction(gt_identify_down , pred_identify_down)
    StatsLogger.instance()["dice_loss_identify"].value = loss_identify.item()
    StatsLogger.instance()["dice_loss_identify_down"].value = loss_identify_down.item()

    loss_identify_ce = ce_lossFunction(pred_identify, gt_identify.float())
    loss_identify_down_ce = ce_lossFunction(pred_identify_down, gt_identify_down.float())

    StatsLogger.instance()["loss_identify_ce"].value = loss_identify_ce.item()
    StatsLogger.instance()["loss_identify_down_ce"].value = loss_identify_down_ce.item()

    loss += (loss_identify + loss_identify_down + loss_identify_ce + loss_identify_down_ce)

    return loss

