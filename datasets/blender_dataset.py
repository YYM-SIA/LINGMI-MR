import os
import math
import torch
from pathlib import Path
from torch.utils.data import Dataset
from geometry.camera import CameraModel
from PIL import Image
import numpy as np
import imageio
import cv2
import torchvision.transforms as tfm
from PIL import Image

class BlenderData:

    def __init__(self, path: str, device = "cuda"):

        self.cam = CameraModel("assets/blender.yaml")
        self.K = self.cam.K.float()
        self.invK = self.cam.IK.float()

        self.near = 0.01  # ratio of focal length
        self.far = 100   # ratio of focal length
        self.focal = 20   # [mm]

        self.log_near = math.log10(self.near)
        self.log_far = math.log10(self.far)

        self.path = path
        self.filenames = os.listdir(os.path.join(path, 'images'))
        self.length = len(self.filenames)
        self.data = {}

        self.transform = tfm.Compose([
            tfm.ToTensor(),
            tfm.Resize(320),
            tfm.CenterCrop(320)
        ])

    def get_index(self, index):
        fname: Path = self.filenames[index]
        return int(fname.stem)

    def get_color(self, index):
        index = index + 1
        color_path = os.path.join(self.path, 'images', '{}.png'.format(str(index).zfill(4)))
        img = cv2.imread(color_path)
        img = np.array(img).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img)
        img = img.unsqueeze(0)
        return img

    def get_depth(self, index):
        index = index + 1
        depth_path = os.path.join(self.path, 'depth', '{}_depth.npy'.format(str(index).zfill(8)))
        depth = np.load(depth_path).astype(np.float32) / self.focal
        depth = torch.from_numpy(depth)[None, None]
        return depth

    def get_intrinsic(self):
        return self.K, self.invK

    def get_data(self, index):
        f = self.filenames[index]
        return torch.load(f)


CASE_NORMAL = 0
CASE_LIGHT = 1
CASE_DARK = 2


class BlenderDataset(Dataset):
    """ Blender sysnthesis dataset.
    """

    def __init__(self, path: str, *args, **kwargs):
        super(BlenderDataset, self).__init__(*args, **kwargs)
        self.src = BlenderData(path)

    def __getitem__(self, index):
        color = self.src.get_color(index)
        depth = self.src.get_depth(index)
        # The shape of color and depth should be [B, C, H, W]
        # and the dtype should be float32
        inputs = {
            "color": color,
            "depth": depth,
        }
        return inputs

    def __len__(self):
        return self.src.length