import torch
import numpy as np
import kornia
import torch.nn.functional as F
import torchvision.transforms.functional as Fn


def pose_loss(Tgt: np.ndarray, Tpred: np.ndarray, points: np.ndarray):
    """
    Xiang Y, Schmidt T, Narayanan V, et al. Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes[J]. arXiv preprint arXiv:1711.00199, 2017.
    """
    assert(Tgt.shape[-2:] == (4, 4))        # B, 4, 4
    assert(Tpred.shape[-2:] == (4, 4))      # B, 4, 4
    assert(points.shape[-2] == 4)           # B, 4, N

    B = Tgt.shape[0]
    N = points.shape[-1]

    p1 = Tgt @ points
    p2 = Tpred @ points
    ploss = torch.mean(torch.norm(p1[:, :3, :] - p2[:, :3, :], dim=1))

    # TODO: use sdf to make SLoss

    return ploss


def smooth_loss(disp: torch.Tensor, img: torch.Tensor,
                mask: torch.Tensor = None, reduce: str = 'mean'):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    assert reduce in ('sum', 'mean', 'norm')
    grad_disp = kornia.filters.sobel(disp).abs()
    grad_img = kornia.filters.sobel(img).abs().mean(1, keepdim=True)
    grad = torch.exp(-grad_img) * grad_disp
    if mask is not None:
        grad = grad[mask]

    if reduce == 'mean':
        loss = (grad * grad).mean()
    elif reduce == 'sum':
        loss = (grad * grad).sum(dim=(-2, -1)).mean()
    elif reduce == 'norm':
        loss = torch.norm(grad, dim=(-2, -1)).mean()
    else:
        raise NotImplementedError()
    return loss


def depth_loss(gt: torch.Tensor, pred: torch.Tensor,
               # min_depth: float = None, max_depth: float = None,
               use_depth: bool = True, use_gradient: bool = True, use_normal: bool = True):
    """
    1. Hu, J., Ozay, M., Zhang, Y. & Okatani, T. Revisiting single image depth estimation: Toward higher resolution maps with accurate object boundaries. Proc. - 2019 IEEE Winter Conf. Appl. Comput. Vision, WACV 2019 1043–1051 (2019) doi:10.1109/WACV.2019.00116.
    Args:
        gt (torch.Tensor): [B,1,H,W]
        pred (torch.Tensor): [B,1,H,W]
        min_depth (float, optional): [description]. Defaults to None.
        max_depth (float, optional): [description]. Defaults to None.
    """

    if not(use_depth or use_gradient or use_normal):
        return 0

    loss = 0
    if use_depth:
        loss_depth = F.l1_loss(pred, gt)
        loss += loss_depth

    if use_gradient or use_normal:
        gt_grad = kornia.filters.spatial_gradient(gt)
        pred_grad = kornia.filters.spatial_gradient(pred)
        gt_grad_dx = gt_grad[:, :, 0].contiguous()
        gt_grad_dy = gt_grad[:, :, 1].contiguous()

        if use_gradient:
            pred_grad_dx = pred_grad[:, :, 0].contiguous()
            pred_grad_dy = pred_grad[:, :, 1].contiguous()
            loss_dx = torch.abs(pred_grad_dx - gt_grad_dx).mean()
            loss_dy = torch.abs(pred_grad_dy - gt_grad_dy).mean()
            loss = loss_normal

        if use_normal:
            ones = torch.autograd.Variable(torch.ones_like(gt))
            gt_normal = torch.cat((-gt_grad_dx, -gt_grad_dy, ones), 1)
            pred_normal = torch.cat((-pred_grad_dx, -pred_grad_dy, ones), 1)
            loss_normal = torch.abs(1 - F.cosine_similarity(pred_normal, gt_normal, dim=1))
            loss_normal = loss_normal.unsqueeze(1).mean()
            loss = loss_depth * use_depth + loss_normal * use_normal + (loss_dx + loss_dy) * use_gradient
            loss += loss_dx + loss_dy

    return loss


def mean_on_mask(diff: torch.Tensor, mask: torch.Tensor, min_size: int = 1000):
    """
    1. Ozyoruk, K. B. et al. EndoSLAM dataset and an unsupervised monocular visual odometry and depth estimation approach for endoscopic videos. Med. Image Anal. 71, (2021).
    """
    k = mask.sum(dim=(-3, -2, -1))
    sel = (k > min_size).squeeze()
    mean_value = (diff[sel] * mask[sel]).sum(dim=(-3, -2, -1)) / k[sel]
    return mean_value.mean()


def brightnes_equator(source, target):
    """
    1. Ozyoruk, K. B. et al. EndoSLAM dataset and an unsupervised monocular visual odometry and depth estimation approach for endoscopic videos. Med. Image Anal. 71, (2021).
    """

    def image_stats(image):
        # compute the mean and standard deviation of each channel

        l = image[:, 0, :, :]
        a = image[:, 1, :, :]
        b = image[:, 2, :, :]

        (lMean, lStd) = (torch.mean(torch.squeeze(l)), torch.std(torch.squeeze(l)))

        (aMean, aStd) = (torch.mean(torch.squeeze(a)), torch.std(torch.squeeze(a)))

        (bMean, bStd) = (torch.mean(torch.squeeze(b)), torch.std(torch.squeeze(b)))

        # return the color statistics
        return (lMean, lStd, aMean, aStd, bMean, bStd)

    def color_transfer(source, target):
        # convert the images from the RGB to L*ab* color space, being
        # sure to utilizing the floating point data type (note: OpenCV
        # expects floats to be 32-bit, so use that instead of 64-bit)

        # compute color statistics for the source and target images
        (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
        (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

        # subtract the means from the target image
        l = target[:, 0, :, :]
        a = target[:, 1, :, :]
        b = target[:, 2, :, :]

        l = l - lMeanTar
        #print("after l",torch.isnan(l))
        a = a - aMeanTar
        b = b - bMeanTar
        # scale by the standard deviations
        l = (lStdTar / lStdSrc) * l
        a = (aStdTar / aStdSrc) * a
        b = (bStdTar / bStdSrc) * b
        # add in the source mean
        l = l + lMeanSrc
        a = a + aMeanSrc
        b = b + bMeanSrc
        transfer = torch.cat((l.unsqueeze(1), a.unsqueeze(1), b.unsqueeze(1)), 1)
        # print(torch.isnan(transfer))
        return transfer

    # return the color transferred image
    transfered_image = color_transfer(target, source)
    return transfered_image
