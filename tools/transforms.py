import numbers
import numpy as np
from typing import List, Optional
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as Fn
from PIL.Image import Image


class ToNumpy:
    """ Randomly change the gamma.
    """

    def __call__(self, pic):
        return np.asarray(pic)


class NpToTensor:
    """ Randomly change the gamma.
    """

    def __call__(self, pic: np.ndarray):
        if isinstance(pic, Image):
            pic = np.asarray(pic)
        return torch.from_numpy(pic).float()


class GammaJitter(nn.Module):
    """ Randomly change the gamma.
    """

    def __init__(self, gamma=0, gain=None):
        super().__init__()
        self.gain = gain
        self.gamma = self._check_input(gamma, 'gamma')

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(gamma: Optional[List[float]]):
        g = None if gamma is None else float(torch.empty(1).uniform_(gamma[0], gamma[1]))
        return g

    def forward(self, img):
        gamma_factor = self.get_params(self.gamma)
        if gamma_factor is not None:
            img = Fn.adjust_gamma(img, gamma_factor)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'gamma={0})'.format(self.gamma)
        return format_string
