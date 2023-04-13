from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraModel(object):

    def __init__(self, path: str):
        """
        Args:
            path (str): Intrinsics file of yaml format.
        """
        from tools.utils import load_yaml
        param = load_yaml(path)
        self.K = torch.FloatTensor(param['intrinsics'])
        self.IK = torch.inverse(self.K)
        self.rd = torch.FloatTensor(param['radial_distortion'])
        self.td = torch.FloatTensor(param['tangent_distortion'])

    def to(self, device: str):
        self.K = self.K.to(device)
        self.IK = self.IK.to(device)
        self.rd = self.rd.to(device)
        self.td = self.td.to(device)
        return self

    def cuda(self):
        self.to('cuda')
        return self

    def cpu(self):
        self.to('cpu')
        return self

    def __call__(self, inverse: bool = False):
        return self.K if not inverse else self.IK


class DepthBackprojD(nn.Module):
    """Layer to transform a depth image into a point cloud.
    """

    def __init__(self, height: int, width: int, IK: torch.Tensor,
                 rdistor: List[float] = None, tdistor: List[float] = None,):
        super(DepthBackprojD, self).__init__()

        H, W = height, width
        v, u = torch.meshgrid(
            torch.linspace(0, 1, H).float(),
            torch.linspace(0, 1, W).float(), indexing='ij')
        i = torch.ones_like(u)

        self.height = height
        self.width = width
        self.uv = torch.stack([u, v, i], -1).view(H, W, 3, 1).to(IK.device)
        self.uv = nn.Parameter(self.uv, requires_grad=False)
        self.f = torch.FloatTensor([1 - 1 / W, 1 - 1 / H, 1.0]).view(1, 1, 3).to(IK.device)
        self.f = nn.Parameter(self.f, requires_grad=False)

        # A. back-projection
        self.IK = IK.view(4, 4).float()
        self.pix = (self.IK[:3, :3] @ self.uv).view(H, W, 3)
        self.pix *= self.f  # u -> u(W-1)/W, v -> v(H-1)/H

        # B. de-distortion
        x, y = self.pix[:, :, 0:1], self.pix[:, :, 1:2]
        r2 = x * x + y * y
        # a). radial distortion
        kr = 1.0
        if rdistor is not None:
            k1, k2, k3 = rdistor
            r4 = r2 * r2
            r6 = r4 * r2
            kr += k1 * r2 + k2 * r4 + k3 * r6
        # b). tangent distortion
        kdx, kdy = 0.0, 0.0
        if tdistor is not None:
            p1, p2 = tdistor
            kdx, kdy = 2 * p1 * x * y + p2 * (r2 + 2 * x * x), p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        # c). total distortion
        self.pix[:, :, 0:1] = x * kr + kdx
        self.pix[:, :, 1:2] = y * kr + kdy
        self.pix = nn.Parameter(self.pix)

    def forward(self, depth: torch.Tensor, ret_scaled_depth=False,
                use_mono: bool = True, dim: int = -3):
        """ 
        Args:
            depth: [B,N,H,W]
            K (torch.Tensor, optional): The *normalized* camera intrinsics matrix [4, 4].
        """
        with torch.no_grad():
            H, W = self.height, self.width
            if depth.shape[-2:] != (H, W):
                depth = F.interpolate(depth, (H, W), mode='bilinear', align_corners=False)

            pc = depth.unsqueeze(-1) * self.pix

        if use_mono:
            I = torch.ones_like(depth).unsqueeze(-1)
            pc = torch.cat((pc, I), dim=-1)

        if dim in (-3, 2):
            pc = pc.permute(0, 1, 4, 2, 3).contiguous()

        if ret_scaled_depth:
            return pc, depth
        else:
            return pc
