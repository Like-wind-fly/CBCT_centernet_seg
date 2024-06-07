import torch

from neural_parts_code.stats_logger import StatsLogger
from neural_parts_code.models.boxDecode import  box_decode


def box_IOU(gt_boxes,gt_dwh,pred_boxes,pred_dwh):
    min_corners, _ = torch.max(torch.stack([gt_boxes[:, :, 0], pred_boxes[:, :, 0]], dim=2), dim=2)
    max_corners, _ = torch.min(torch.stack([gt_boxes[:, :, 1], pred_boxes[:, :, 1]], dim=2), dim=2)

    intersect_dwh = max_corners - min_corners
    intersect_dwh[intersect_dwh < 0] = 0
    intersect_area = intersect_dwh[:, 0] * intersect_dwh[:, 1] * intersect_dwh[:, 2]

    gt_area = gt_dwh[:, 0] * gt_dwh[:, 1] * gt_dwh[:, 2]
    pred_area = pred_dwh[:, 0] * pred_dwh[:, 1] * pred_dwh[:, 2]
    IOU = intersect_area * 1.0 / (gt_area + pred_area - intersect_area)

    return  IOU

def box_OIR(gt_boxes,gt_dwh, gt_class,gt_seg,pred_boxes,pred_dwh):
    min_corners, _ = torch.max(torch.stack([gt_boxes[:, :, 0], pred_boxes[:, :, 0]], dim=2), dim=2)
    max_corners, _ = torch.min(torch.stack([gt_boxes[:, :, 1], pred_boxes[:, :, 1]], dim=2), dim=2)
    min_corners = min_corners.int()
    max_corners = max_corners.int()
    gt_boxes =gt_boxes.int()

    intersect_dwh = max_corners - min_corners
    intersect_dwh[intersect_dwh < 0] = 0
    intersect_area = intersect_dwh[:, 0] * intersect_dwh[:, 1] * intersect_dwh[:, 2]

    oir = torch.zeros(gt_class.shape).cuda()
    for i,area in enumerate(intersect_area):

        if area>0:
            class_i = gt_class[i]
            union_roi = gt_seg[0,0,min_corners[i][0]:max_corners[i][0],min_corners[i][1]:max_corners[i][1],min_corners[i][2]:max_corners[i][2]]
            gt_roi    = gt_seg[0, 0, gt_boxes[i][0][0]:gt_boxes[i][0][1], gt_boxes[i][1][0]:gt_boxes[i][1][1],gt_boxes[i][2][0]:gt_boxes[i][2][1]]
            union_roi = union_roi == class_i
            gt_roi = gt_roi == class_i
            oir[i] = union_roi.sum()/gt_roi.sum()
    return  oir

def AP(predictions, targets):
    downSize = 4

    pred_hm = predictions["heatmap"]
    pred_reg = predictions["reg"]
    pred_dwh = predictions["dwh"]

    pred_center, pred_dwh, pred_scores, pred_clses = box_decode(pred_hm, pred_reg, pred_dwh, centerMode=True)
    batch, boxs, _ = pred_dwh.shape

    teethId = torch.zeros(batch, boxs, 1).cuda()
    for i in range(batch):
        teethId[i] = i * 1000
    pred_center = torch.cat([pred_center, teethId], dim=-1)

    pred_center = pred_center.view(-1,4)
    pred_dwh = pred_dwh.view(-1, 3)
    # pred_scores = pred_scores.view(-1)
    # pred_clses = pred_clses.view(-1)
    B,C = pred_clses.shape


    pred_boxes = torch.stack([pred_center[:, :3] - pred_dwh / 2, pred_center[:, :3] + pred_dwh / 2], dim=2)

    gt_mask = targets["reg_mask"]
    gt_index = targets["index_int"]
    gt_reg = targets["reg"]
    gt_dwh = targets["dwh"]
    gt_clses = torch.range(1,32).cuda()[None].repeat(batch,1)

    gt_center = (gt_index + gt_reg) * downSize
    teethId = torch.zeros(batch, 32, 1).cuda()
    for i in range(batch):
        teethId[i] = i * 1000
    gt_center = torch.cat([gt_center, teethId], dim=-1)

    gt_center = gt_center[gt_mask]
    gt_dwh = gt_dwh[gt_mask]

    gt_clses = gt_clses[gt_mask]
    gt_boxes = torch.stack([gt_center[:, :3] - gt_dwh / 2, gt_center[:, :3] + gt_dwh / 2], dim=2)

    N1, _ = gt_center.shape
    N2, _ = pred_center.shape

    diff = gt_center[:, None] - pred_center[None]
    dists = torch.square(diff).sum(-1).sqrt()

    pred_to_gt, gt_id = dists.min(0)

    IOU = box_IOU(gt_boxes[gt_id], gt_dwh[gt_id], pred_boxes, pred_dwh)

    pred_hited = IOU > 0.5

    AP_line = []
    mask = torch.zeros(B, C).cuda().bool()
    recall_32 = 0
    for i in range(1,C+1):
        mask[:,:i] = 1
        pred_hited_i = pred_hited[mask.view(-1)]
        hit_object = gt_id[mask.view(-1)][pred_hited_i]
        unique_hit_object = torch.unique(hit_object)

        TP = hit_object.shape[0]
        FN = N1 - unique_hit_object.shape[0]

        precision = TP /i
        # recall = TP / (TP + FN)
        recall = unique_hit_object.shape[0]/N1
        if i == 32:
            recall_32 = recall
        AP_line.append([precision,recall])

    AP = 0
    st_recall = 0
    for precision,recall in AP_line:
        AP += (recall - st_recall) * precision
        st_recall = recall
    StatsLogger.instance()["AP"].value = AP
    StatsLogger.instance()["recall_32"].value = recall_32


    return AP


def object_detection_metrics(predictions, targets):
    AP(predictions, targets)
    downSize = 4

    pred_hm = predictions["heatmap"]
    pred_reg = predictions["reg"]
    pred_dwh = predictions["dwh"]

    pred_center, pred_dwh, pred_scores, pred_clses = box_decode(pred_hm, pred_reg, pred_dwh, centerMode= True)
    batch, boxs, _ = pred_dwh.shape

    pred_mask = pred_scores > 0.5

    teethId = torch.zeros(batch,boxs,1).cuda()
    for i in range(batch):
        teethId[i] = i * 1000
    pred_center = torch.cat([pred_center,teethId],dim= -1)

    pred_center = pred_center[pred_mask]

    pred_dwh = pred_dwh[pred_mask]
    pred_clses  = pred_clses[pred_mask]

    pred_boxes = torch.stack([pred_center[:,:3] -pred_dwh / 2, pred_center[:,:3] + pred_dwh / 2], dim=2)

    gt_mask  = targets["reg_mask"]
    gt_index = targets["index_int"]
    gt_reg   = targets["reg"]
    gt_dwh   = targets["dwh"]
    gt_clses = torch.range(1,32).cuda()[None].repeat(batch,1)
    gt_center = (gt_index + gt_reg) * downSize
    teethId = torch.zeros(batch,32,1).cuda()
    for i in range(batch):
        teethId[i] = i * 1000
    gt_center = torch.cat([gt_center,teethId],dim= -1)

    gt_center = gt_center[gt_mask]
    gt_dwh  = gt_dwh[gt_mask]

    gt_clses = gt_clses[gt_mask]
    gt_boxes = torch.stack([gt_center[:,:3] - gt_dwh / 2, gt_center[:,:3] + gt_dwh / 2], dim=2)

    N1 , _ = gt_center.shape
    N2 , _ = pred_center.shape

    diff = gt_center[:,None] - pred_center[None]
    dists = torch.square(diff).sum(-1).sqrt()


    pred_to_gt, gt_id = dists.min(0)
    # gt_to_pred, pred_id = dists.min(1)

    # gt_hited = box_IOU(gt_boxes,gt_dwh,pred_boxes[pred_id],pred_dwh[pred_id])
    IOU = box_IOU(gt_boxes[gt_id], gt_dwh[gt_id], pred_boxes, pred_dwh)
    OIR = box_OIR(gt_boxes[gt_id], gt_dwh[gt_id], gt_clses[gt_id], targets["seg"], pred_boxes, pred_dwh)

    pred_hited = IOU > 0.5
    OIR = OIR[pred_hited]

    hit_object = gt_id[pred_hited]
    unique_hit_object = torch.unique(hit_object)

    TP = hit_object.shape[0]
    FN = N1 - unique_hit_object.shape[0]

    precision = 0
    if N2 != 0:
        precision = TP / N2

    recall    = TP / (TP+FN)
    F1score  = 2*(precision*recall)/(precision+recall+1e-18)

    StatsLogger.instance()["precision"].value = precision
    StatsLogger.instance()["recall"].value    = recall
    StatsLogger.instance()["F1score"].value =   F1score
    StatsLogger.instance()["mean_distance"].value = pred_to_gt.mean().item()
    StatsLogger.instance()["avg_IOU"].value = IOU.mean().item()
    StatsLogger.instance()["avg_OIR"].value = OIR.mean().item()
    return 0

if __name__ == "__main__":
    from neural_parts_code.utils import sphere_mesh
    from neural_parts_code.datasets.cbct.common import save_img
    import igl



    shape = torch.tensor([64, 64, 64]).int().cuda()
    rasterizer = Rasterize(shape)
    B = 1

    torch.cuda.set_device("cuda:1")
    vertices, faces = sphere_mesh(n=2000, radius=1,device="cuda:1")
    vertices = vertices[None].expand(B, -1, -1)
    faces = faces[None].expand(B, -1, -1)


    index = vertices.device.index
    pred_voxels_rasterized = rasterizer(vertices, faces).cuda()


    igl.write_obj("sphere.obj",vertices[0].cpu().numpy(), faces[0].cpu().numpy())
    save_img(pred_voxels_rasterized[0].cpu().numpy(),"sphere.nii")


