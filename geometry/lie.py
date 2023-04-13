from typing import List
import torch
import torch.nn as nn
import pytorch3d.transforms as tf3d


def se3_action(G: torch.Tensor, P: torch.Tensor):
    B, N1, _ = G.shape
    B, N2, _, H, W = P.shape
    T = se3_to_matrix(G).view(B, N1, 4, 4)
    return (T @ P.view(B, N2, 4, -1)).view(B, -1, 4, H, W)


def se3_inv(G: torch.Tensor):
    return se3_from_matrix(se3_to_matrix_inv(G))


def se3_from_matrix(mat: torch.Tensor):
    B, N, _, _ = mat.shape
    return tf3d.se3_log_map(mat.view(-1, 4, 4).transpose(-2, -1)).view(B, N, 6)


def se3_to_matrix(G: torch.Tensor):
    B, N, _ = G.shape
    return tf3d.se3_exp_map(G.view(-1, 6)).transpose(-2, -1).view(B, N, 4, 4)


def se3_to_matrix_inv(G: torch.Tensor):
    T = se3_to_matrix(G)
    Rt = T[:, :, :3, :3].transpose(-2, -1).contiguous()
    T[:, :, :3, :3] = Rt
    T[:, :, :3, 3:] = - Rt @ T[:, :, :3, 3:]
    return T


def se3_left_mul(G: torch.Tensor, dG: torch.Tensor):
    """ Update se(3) vector with dG (left multiply)
    """
    # Attention that output of se3_exp_map is transposed
    T = se3_to_matrix(G)
    dT = se3_to_matrix(dG)
    T = dT @ T
    return se3_from_matrix(T)


def to_hat(p: torch.Tensor):
    B, N, C = p.shape
    assert C == 3
    x, y, z = p.unbind(-1)
    o = torch.zeros_like(x)
    h = torch.stack([
        o, -z, y,
        z, o, -x,
        -y, x, o,
    ], dim=-1).view(B, N, 3, 3)
    return h


def adj_matrix(T: torch.Tensor, w: torch.Tensor = None):
    B, N, C1, C2 = T.shape
    assert C1 == C2
    if C1 == 3:
        adj = T
    elif C1 == 4:
        R = T[:, :, :3, :3]
        tx = to_hat(T[:, :, :3, 3])
        adj = torch.zeros((B, N, 6, 6), device=T.device, dtype=T.dtype)
        adj[:, :, :3, :3] = R
        adj[:, :, :3, 3:] = tx@R
        adj[:, :, 3:, 3:] = R
    else:
        raise NotImplementedError()

    if w is not None:
        w = (adj @ w.unsqueeze(-1)).squeeze(-1)
        return w
    return adj


def se3_adj(G: torch.Tensor, b: torch.Tensor = None):
    T = se3_to_matrix(G)
    adj = adj_matrix(T)
    if b is not None:
        adj = torch.einsum("bnxy,bnhwzy->bnhwzx", adj, b)
        return adj

    return adj


def se3_adjT(G: torch.Tensor, b: torch.Tensor = None):
    T = se3_to_matrix(G)
    adj = adj_matrix(T)
    if b is not None:
        adj = torch.einsum("bnhwxy,bnyz->bnhwxz", b, adj)
        return adj
    return adj


class LieOp(nn.Module):
    def __init__(self, size: List[int] = (1, 1), X0=None):
        super(LieOp, self).__init__()
        if X0 is None:
            self.X = nn.Parameter(torch.zeros((*size, 6), dtype=torch.float32))
        else:
            self.X = nn.Parameter(torch.FloatTensor(X0).view((*size, 6)))
        self.register_parameter('x', self.X)

    def forward(self, P: torch.Tensor):
        return se3_action(self.X, P)

    def matrix(self):
        return se3_to_matrix(self.X)

    def matrix_inv(self):
        return se3_to_matrix_inv(self.X)
