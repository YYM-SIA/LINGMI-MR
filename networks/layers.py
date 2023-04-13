import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.nn.quantized as Q
import torchvision.transforms.functional as Fn
from collections import OrderedDict
from typing import Callable, Optional


def get_param_count(model: nn.Module, verbose=0):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    total = num_params / 1e6
    if verbose > 1:
        print(model)
    if verbose > 0:
        print('%s: %.3f M' % (model._get_name(), total))
    return total


def disp_normalize(disp: torch.Tensor):
    return disp / disp.mean(dim=(-2, -1), keepdim=True)


def disp_to_depth(disp: torch.Tensor, min_depth: float, max_depth: float, norm: bool = True):
    if norm:
        disp = disp_normalize(disp)
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    inv_depth = min_disp + (max_disp - min_disp) * disp
    depth = 1 / inv_depth
    return inv_depth, depth


def disp_to_depth_log10(disp: torch.Tensor, logmin_depth: float, logmax_depth: float,
                        focal: float = 1.0, norm: bool = False):
    if norm:
        disp = disp_normalize(disp)
    scaled_depth = logmin_depth + (logmax_depth - logmin_depth) * disp
    depth = torch.pow(10, scaled_depth) * focal
    return scaled_depth, depth


def init_weights(net: nn.Module, init_type='xavier', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


################################################################
# 2D Modules
################################################################

def convbn(in_planes, out_planes, kernel, stride=1, norm=None, activ=None, groups=1, bias=None, pad=None, **kwargs):
    if bias is None:
        bias = norm is not None and not isinstance(norm, nn.modules.batchnorm._BatchNorm)
    if pad is None:
        pad = kernel // 2
    return nn.Sequential(OrderedDict([
        ("conv2", nn.Conv2d(in_planes, out_planes, kernel_size=kernel,
                            stride=stride, padding=pad, bias=bias,
                            groups=groups, **kwargs)),
        ("bn2", nn.BatchNorm2d(out_planes) if norm is None else norm),
        ("relu", nn.LeakyReLU(inplace=True) if activ is None else activ)
    ]))


def deconvbn_2d(in_planes, out_planes, kernel, stride=1, norm=None, activ=None, groups=1, bias=None, pad=None, **kwargs):
    if bias is None:
        bias = norm is not None and not isinstance(norm, nn.modules.batchnorm._BatchNorm)
    if pad is None:
        pad = kernel // 2
    return nn.Sequential(OrderedDict([
        ("deconv2", nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel,
                                       stride=stride, padding=pad,
                                       output_padding=pad, bias=bias, groups=groups, **kwargs)),
        ("bn2", nn.BatchNorm2d(out_planes) if norm is None else norm),
        ("relu", nn.LeakyReLU(inplace=True) if activ is None else activ)
    ]))


def upconvbn_2d(scale, in_planes, out_planes, kernel, norm=None, activ=None, groups=1, bias=None, pad=None, **kwargs):
    if bias is None:
        bias = norm is not None and not isinstance(norm, nn.modules.batchnorm._BatchNorm)
    if pad is None:
        pad = kernel // 2
    return nn.Sequential(OrderedDict([
        ("upsample", nn.Upsample(scale_factor=scale, mode='bilinear',
                                 align_corners=False, groups=groups)),
        ("deconv2", nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=1,
                              padding=pad, bias=bias, **kwargs)),
        ("bn2", nn.BatchNorm2d(out_planes) if norm is None else norm),
        ("relu", nn.LeakyReLU(inplace=True) if activ is None else activ)
    ]))


class ConvGRU(nn.Module):
    def __init__(self, h_planes=128, i_planes=128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(h_planes + i_planes, h_planes, 3, padding=1)
        self.convr = nn.Conv2d(h_planes + i_planes, h_planes, 3, padding=1)
        self.convq = nn.Conv2d(h_planes + i_planes, h_planes, 3, padding=1)

    def forward(self, net: torch.Tensor, *inputs):
        inp = torch.cat(inputs, dim=1)
        net_inp = torch.cat([net, inp], dim=1)

        z = torch.sigmoid(self.convz(net_inp))
        r = torch.sigmoid(self.convr(net_inp))
        q = torch.tanh(self.convq(torch.cat([r * net, inp], dim=1)))

        net = (1 - z) * net + z * q
        return net


class ResidualConv2d(nn.Module):

    expansion: int = 1

    def __init__(self, inplanes: int, outplanes: int,
                 stride: int = 1, kernel: int = 3, groups: int = 1,
                 downsample: Optional[nn.Module] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 actv_layer: Optional[Callable[..., nn.Module]] = None):
        super(ResidualConv2d, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if actv_layer is None:
            actv_layer = nn.LeakyReLU
        self.layer1 = convbn(inplanes, outplanes, kernel=kernel, stride=stride, groups=groups,
                             norm=norm_layer(outplanes), activ=actv_layer(inplace=True))
        self.layer2 = convbn(outplanes, outplanes, kernel=kernel, stride=1, groups=groups,
                             norm=norm_layer(outplanes), activ=nn.Identity())
        self.downsample = downsample
        self.relu = actv_layer(inplace=True)

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.layer1(x)
        out = self.layer2(out)
        if self.downsample is not None:
            out += self.downsample(identity)
        out = self.relu(out)
        return out


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.LeakyReLU,
        dilation: int = 1,
        inplace: bool = True,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                            dilation=dilation, groups=groups, bias=norm_layer is None)]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
            self.bn_info = f'BN={layers[-1]._get_name()}'
        else:
            self.bn_info = ''

        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
            self.act_info = f'ACT={layers[-1]._get_name()}'
        else:
            self.act_info = ''

        super().__init__(*layers)
        self.out_channels = out_channels
        self.apply(init_weights)

    def __repr__(self):
        # Only main str
        lines = []
        # info of conv
        extra_repr = self[0].extra_repr()
        if extra_repr:
            lines = extra_repr.split('\n')
        # info of BN and act
        lines.append(f"    {self.bn_info},  {self.act_info}")

        main_str = self._get_name() + '('
        if lines:
            main_str += '\n'.join(lines)
        main_str += ')'
        return main_str

################################################################
# 3D Modules
################################################################


def convbn_3d(in_planes, out_planes, kernel, stride=1, norm=None, activ=None, groups=1, bias=None, pad=None, **kwargs):
    if bias is None:
        bias = norm is not None and not isinstance(norm, nn.modules.batchnorm._BatchNorm)
    if pad is None:
        pad = kernel // 2
    return nn.Sequential(OrderedDict([
        ("conv3", nn.Conv3d(in_planes, out_planes, kernel_size=kernel,
                            stride=stride, padding=pad, bias=bias, groups=groups, **kwargs)),
        ("bn3", nn.BatchNorm3d(out_planes) if norm is None else norm),
        ("relu", nn.LeakyReLU(inplace=True) if activ is None else activ)
    ]))


def deconvbn_3d(in_planes, out_planes, kernel, stride=1, norm=None, activ=None, groups=1, bias=None, pad=None, **kwargs):
    if bias is None:
        bias = norm is not None and not isinstance(norm, nn.modules.batchnorm._BatchNorm)
    if pad is None:
        pad = kernel // 2
    return nn.Sequential(OrderedDict([
        ("deconv3", nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel,
                                       stride=stride, padding=pad,
                                       output_padding=pad, bias=bias, groups=groups, **kwargs)),
        ("bn3", nn.BatchNorm3d(out_planes) if norm is None else norm),
        ("relu", nn.LeakyReLU(inplace=True) if activ is None else activ)
    ]))


def upconvbn_3d(scale, in_planes, out_planes, kernel, norm=None, activ=None, groups=1, bias=None, pad=None, **kwargs):
    if bias is None:
        bias = norm is not None and not isinstance(norm, nn.modules.batchnorm._BatchNorm)
    if pad is None:
        pad = kernel // 2
    return nn.Sequential(OrderedDict([
        ("upsample", nn.Upsample(scale_factor=scale, mode='bilinear')),
        ("deconv2", nn.Conv2d(in_planes, out_planes, kernel_size=kernel,
                              stride=1, padding=pad, bias=bias, groups=groups, **kwargs)),
        ("bn2", nn.BatchNorm3d(out_planes) if norm is None else norm),
        ("relu", nn.LeakyReLU(inplace=True) if activ is None else activ)
    ]))
