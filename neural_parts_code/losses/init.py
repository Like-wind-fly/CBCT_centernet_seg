import torch

from ..stats_logger import StatsLogger
from .centernet_loss import heatmap_loss,offset_loss,size_loss,class_ce_loss,heatmap32_loss
from .SGA_loss import tm_SGA_loss
from .SEG_loss import tm_quad_loss

losses = {


        "heatmap_loss": heatmap_loss,
        "heatmap32_loss": heatmap32_loss,
        "offset_loss": offset_loss,
        "size_loss": size_loss,
        "class_ce_loss": class_ce_loss,
        "tm_quad_loss":tm_quad_loss,
        "tm_SGA_loss" : tm_SGA_loss,
    }

def get_loss(loss_types, config):
    return weighted_sum_of_losses(
        *[losses[t] for t in loss_types],
        weights=config["weights"]
    )

def eval_loss(loss_types, config):
    return gradient_of_losses(
        *[losses[t] for t in loss_types],
        weights=config["weights"]
    )



def weighted_sum_of_losses(*functions, weights=None):
    if weights is None:
        weights = [1.0]*len(functions)
    def inner(*args, **kwargs):
        weights = args[2]["weights"]
        losses = []
        for w, loss_fn in zip(weights, functions):
            if w != 0:
                ll = w*loss_fn(*args, **kwargs)
                if ll.device.type != "cpu" and not torch.isnan(ll):
                    losses.append(ll)
        if len(losses) == 0:
            loss = torch.tensor(1).float().cuda()
            loss.requires_grad_(True)
            return loss
        return sum(losses)
    return inner

def gradient_of_losses(*functions, weights=None):
    if weights is None:
        weights = [1.0]*len(functions)

    def inner(*args, **kwargs):
        x = torch.tensor(args[0]["y_prim"].detach(), requires_grad=True)
        args[0]["y_prim"] = x
        y_prim = args[0]["y_prim"]
        gradient = []

        for w, loss_fn in zip(weights, functions):
            loss = w * loss_fn(*args, **kwargs)
            loss.backward()
            gradient.append(y_prim.grad.detach().clone())
            y_prim.grad.zero_()

        gradient = torch.stack(gradient,dim=0)
        return gradient

    return inner

def size_regularizer(predictions, target, config):
    prim_points = predictions["y_prim"]

    # Compute the pairwise square dists between points on a primitive
    loss = ((prim_points[:, :, None] - prim_points[:, None])**2).sum(-1).mean()

    StatsLogger.instance()["size"].value = loss.item()

    return loss


def non_overlapping_regularizer(predictions, target, config):
    max_shared_points = config.get("max_shared_points", 2)
    sharpness = config.get("sharpness", 1.0)

    phi_volume = predictions["phi_volume"]
    # Compute the inside/outside function based on these distances
    primitive_inside_outside = torch.sigmoid(sharpness * (-phi_volume))
    loss = (primitive_inside_outside.sum(-1) - max_shared_points).relu().mean()

    StatsLogger.instance()["overlapping"].value = loss.item()

    return loss

if __name__ == "__main__":
    pass




