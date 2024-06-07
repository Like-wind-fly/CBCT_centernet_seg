import torch

from neural_parts_code.stats_logger import StatsLogger
from neural_parts_code.PostProcess.SGA_visualize import get_meshes
from neural_parts_code.datasets.cbct.common import save_img

import torch
from pytorch3d.ops import sample_points_from_meshes, knn_points, knn_gather

from pytorch3d.structures import Meshes
from neural_parts_code.stats_logger import StatsLogger

#       2|P∩G|
# DSC = --------
#       |P|+|G|
def SGA_dsc(predictions, targets):
    pred_seg = predictions["volume"]
    gt_seg = targets["seg"]


    AND = (((pred_seg == gt_seg) * pred_seg)>0).sum(dim=[1, 2, 3, 4])
    P = (pred_seg>0).sum(dim=[1, 2, 3, 4])
    G = (gt_seg>0).sum(dim=[1, 2, 3, 4])

    value = (2.0*AND/(P+G)).mean().item()

    StatsLogger.instance()["dsc"].value = value
    StatsLogger.instance()["seg_precision"].value = (AND/P).mean().item()
    StatsLogger.instance()["seg_recall"].value = (AND / G).mean().item()

    return value



def evaluate_mesh(predictions, targets):
    x_classify,x_gt_meshes,x_pred_meshes = get_meshes(predictions, targets)

    pred_verts = x_pred_meshes[0][0]
    pred_faces = x_pred_meshes[0][1]

    gt_verts = x_gt_meshes[0][0]
    gt_faces = x_gt_meshes[0][1]

    meshes = Meshes(verts=gt_verts+pred_verts, faces=gt_faces+pred_faces)
    pcl_points, pcl_normals = sample_points_from_meshes(meshes, num_samples= 3000, return_normals=True)

    length = len(pcl_points)

    gt_points, gt_normals = pcl_points[0:int(length/2)], pcl_normals[0:int(length/2)]
    pred_points, pred_normals = pcl_points[int(length/2):], pcl_normals[int(length/2):]
    torch.save([gt_points,pred_points],"sample.pt")

    # compute metrics
    metrics = compute_metrics(gt_points, gt_normals, pred_points, pred_normals)
    # metrics['SI'] = SI(meshes[length/2:])

    for k in metrics.keys():
        StatsLogger.instance()[k].value = metrics[k]
    return


def compute_metrics(gt_pcl, gt_normals, pred_pcl, pred_normals):
    # transform in batches
    gt_pcl = gt_pcl.unsqueeze(0) if gt_pcl.ndim == 2 else gt_pcl
    gt_normals = gt_normals.unsqueeze(
        0) if gt_normals.ndim == 2 else gt_normals
    pred_pcl = pred_pcl.unsqueeze(0) if pred_pcl.ndim == 2 else pred_pcl
    pred_normals = pred_normals.unsqueeze(
        0) if pred_normals.ndim == 2 else pred_normals

    # compute metrics
    metrics = {}
    # For each predicted point, find its neareast-neighbor GT point
    knn_p2g, knn_g2p = knn_points(pred_pcl, gt_pcl, K=1), knn_points(gt_pcl, pred_pcl, K=1)
    knn_p2g_sq_dists, knn_g2p_sq_dists = knn_p2g.dists.squeeze(dim=-1), knn_g2p.dists.squeeze(dim=-1)

    knn_p2g_dists, knn_g2p_dists = knn_p2g_sq_dists.sqrt(), knn_g2p_sq_dists.sqrt()

    metrics['CHAMFER'] = (knn_p2g_dists.mean(dim=-1) + knn_g2p_dists.mean(dim=-1)).mean().item()/2
    metrics['HAUSDORFF'] = torch.stack([knn_p2g_dists.max(dim=-1).values, knn_g2p_dists.max(dim=-1).values], dim=1).max(dim=-1).values.mean().item()
    metrics['HAUSDORFF_95'] = torch.stack(
        [torch.quantile(knn_p2g_dists, 0.95, dim=-1), torch.quantile(knn_g2p_dists, 0.95, dim=-1)], dim=1).max(dim=-1).values.mean().item()

    gt_noramls_near_pred = knn_gather(gt_normals, knn_p2g.idx)[..., 0, :]
    pred_to_gt_dot = (pred_normals * gt_noramls_near_pred).sum(dim=-1).abs().mean(dim=-1)
    pred_normals_near_gt = knn_gather(pred_normals, knn_g2p.idx)[..., 0, :]
    gt_to_pred_dot = (gt_normals * pred_normals_near_gt).sum(dim=-1).abs().mean(dim=-1)
    metrics['NORMAL_CONSISTENCY'] = (0.5 * (pred_to_gt_dot + gt_to_pred_dot)).mean().item()
    return metrics


