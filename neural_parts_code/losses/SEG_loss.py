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
            dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        # 返回的是dice距离
        dice = dice / pred.size(1)
        loss = (1 - dice)
        meanloss = loss.mean()
        return meanloss

def tm_quad_loss(predictions, targets, config):
    gt_seg = targets["quad_seg"]
    B,C,W,H,D = gt_seg.shape
    gt_seg = gt_seg.view(B*C,W,H,D).unsqueeze(dim=1)
    gt_seg = torch.cat([gt_seg == i for i in range(1, 5)], dim=1)

    pred_seg = predictions["quad_seg"]
    # pred_seg = nn.Softmax(pred_seg)

    # print(gt_seg.shape)
    # print(pred_seg.shape)




    lossFunction = DiceLoss()
    loss_seg = lossFunction(gt_seg, pred_seg)

    StatsLogger.instance()["dice_loss_seg"].value = loss_seg.item()
    loss = loss_seg

    return loss

