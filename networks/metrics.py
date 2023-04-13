import torch
import numpy as np


def compute_pose_metrics(g1: torch.Tensor, g2: torch.Tensor):
    """ Translation/Rotation metrics from se(3) """
    t1, q1 = g1.split([3, 3], -1)
    t2, q2 = g2.split([3, 3], -1)

    # convert radians to degrees
    r_err = (q1-q2).norm(dim=-1)
    t_err = (t1-t2).norm(dim=-1)
    return r_err, t_err


def compute_depth_metrics(gt: torch.Tensor, pred: torch.Tensor):
    """
    # Accuracy Measures #
    Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return {
        "de/abs_rel": abs_rel,
        "de/sq_rel": sq_rel,
        "de/rms": rmse,
        "de/log_rms": rmse_log,
        "da/a1": a1,
        "da/a2": a2,
        "da/a3": a3,
    }
