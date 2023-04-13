import torch
import torch.nn as nn
import torch.nn.functional as F
from math import inf
from geometry.lie import se3_action


class Projector(nn.Module):
    """Layer to transform a point cloud to a image coordinates.
    """

    def forward(self, P0: torch.Tensor, K: torch.Tensor, G: torch.Tensor = None,
                norm: bool = True,
                ret_point_cloud: bool = False,
                min_depth: float = 1e-6,
                max_depth: float = inf,
                use_lie: bool = True,
                split: bool = False):
        """[summary]

        Args:
            P0 (torch.Tensor): [B,N,4,H,W]
            K (torch.Tensor): The *normalized* camera intrinsics matrix [4, 4].
            G (torch.Tensor, optional): [B,[N,]6]
            norm: Unused, should always True.
            ret_point_cloud (bool, optional): If return intermediate transformed point cloud P1.
            min_depth (float, optional): Clipped minimum depth.
            max_depth (float, optional): Clipped maximum depth.
            use_lie (bool, optional): G is se(3) [use_lie=True] or SE(3) [use_lie=False].
            split (bool, optional): If split pixel coordinates and depth channel.

        Returns:
            tuple: uv: pixel coordinates [B,N,C,H,W]
                   [d]: If split, depth [B,N,1,H,W]
                   [P1]: If ret_point_cloud, [B,N,4,H,W]
        """
        assert(norm)
        with torch.no_grad():
            B, N1, _, H, W = P0.shape

            if G is not None:
                if use_lie:
                    NG = 1 if G.dim() == 2 else N1
                    G = G.view(B, NG, 6)
                    P1 = se3_action(G, P0)
                else:
                    NG = 1 if G.dim() == 3 else N1
                    G = G.view(B, NG, 4, 4)
                    P1 = torch.einsum('bnij,bnjhw->bnihw', G, P0)
            else:
                P1 = P0

            K = K.view(4, 4).clone()
            # Fix intrinsics with width and height
            K[0, 0] *= W / (W-1)
            K[1, 1] *= H / (H-1)
            pixel = torch.einsum("ij,bnjhw->bnihw", K[:3], P1)

            z = torch.clamp(pixel[:, :, 2:3], 0, max_depth)
            iz = torch.where(z.abs() < min_depth, torch.zeros_like(z), 1.0 / z)

            uv = pixel[:, :, :2] * iz
            uv = uv * 2 - 1.0  # [0,1] to [-1,1]

            ret_list = []
            if not split:
                uv = torch.cat([uv, z], dim=-3)
            ret_list.append(uv)
            if split:
                ret_list.append(z)
            if ret_point_cloud:
                ret_list.append(P1)
            return tuple(ret_list)


class DepthBackproj(nn.Module):
    """Layer to transform a depth image into a point cloud.
    """

    def __init__(self, height: int, width: int):
        super(DepthBackproj, self).__init__()

        H, W = height, width
        v, u = torch.meshgrid(
            torch.linspace(0, 1, H).float(),
            torch.linspace(0, 1, W).float(), indexing='ij')
        i = torch.ones_like(u)

        self.height = height
        self.width = width
        self.uv = torch.stack([u, v, i], -1).view(H, W, 3, 1)
        self.uv = nn.Parameter(self.uv, requires_grad=False)
        self.f = torch.FloatTensor([1 - 1 / W, 1 - 1 / H, 1.0]).view(1, 1, 3)
        self.f = nn.Parameter(self.f, requires_grad=False)

    def forward(self, depth: torch.Tensor, IK: torch.Tensor,
                norm: bool = True, ret_scaled_depth=False,
                use_mono: bool = True, dim: int = -3):
        """ 
        Args:
            K (torch.Tensor, optional): The *normalized* camera intrinsics matrix [4, 4].
            use_mono (bool, optional): If use monogeneous vector mode, i.e. pc in shape [B,N,4,H,W]
        """
        assert(norm)
        assert dim in (-3, -1, 2, 4)

        H, W = self.height, self.width
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(depth, (H, W), mode='bilinear', align_corners=False)

        IK = IK.view(4, 4)
        pix = (IK[:3, :3] @ self.uv).view(H, W, 3)
        pix *= self.f  # u -> u(W-1)/W, v -> v(H-1)/H
        pc = depth.unsqueeze(-1) * pix

        if use_mono:
            I = torch.ones_like(depth).unsqueeze(-1)
            pc = torch.cat((pc, I), dim=-1)

        if dim in (-3, 2):
            pc = pc.permute(0, 1, 4, 2, 3).contiguous()

        if ret_scaled_depth:
            return pc, depth
        else:
            return pc
