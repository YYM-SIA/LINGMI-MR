import os
import re
import glob
import numpy as np
import PIL.Image as pil
import torch
from torchvision.datasets.folder import pil_loader
from datasets.mono_dataset import MonoDataset
from geometry.camera import CameraModel
import pytorch3d.transforms as tf3d


class EndoSlamDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """

    def __init__(self, drop_last=True, camera: str = None, *args, **kwargs):
        super(EndoSlamDataset, self).__init__(*args, **kwargs)

        if camera is None:
            self.cam = CameraModel('assets/endoslam-unity.yaml')
        else:
            self.cam = CameraModel(camera)

        self.K = self.cam.K
        self.invK = self.cam.IK

        self.near = 0.01  # ratio of focal length
        self.far = 100   # ratio of focal length
        self.log_near = np.log10(self.near)
        self.log_far = np.log10(self.far)

        # Replace self.length for bidirection indexing
        self.load_poses_csv()

        self.load_dataset()
        self.filenames = self.filenames[:self.length]
        
        self.length = min(len(self.poses), len(self.filenames))

    def load_dataset(self):
        # Load image file names to id
        self.filenames = []
        folder = os.path.join(self.data_path, 'Frames')
        assert(os.path.exists(folder))
        if self.split == 'all':
            for file in os.listdir(folder):
                if file.endswith(self.img_ext):
                    self.filenames.append(file.replace(self.img_ext, ""))
            import re
            self.filenames.sort(key=lambda f: int(re.sub(r'\D', '', f)))
        else:
            file = os.path.join(self.data_path, "{}.txt".format(self.split))
            assert(os.path.exists(file))
            with open(file, 'r') as f:
                self.filenames = f.readlines()

    def load_poses_csv(self):
        pose_folder = os.path.join(self.data_path, 'Poses/*.csv')
        pose_files = glob.glob(pose_folder)
        self.poses = np.loadtxt(pose_files[0], delimiter=',', skiprows=1)

    def get_index(self, index):
        fname = self.filenames[index]
        match = re.match(r"image_(\d+)", fname)
        id = int(match[1])
        return id

    def get_color(self, index):
        file = "image_%04d.png" % index
        color = self.loader(os.path.join(self.data_path, 'Frames', file))
        return color

    def get_depth(self, index):
        file = "aov_image_%04d.png" % index
        img = pil.open(os.path.join(self.data_path, 'Pixelwise Depths', file)).convert("L")
        img = np.asarray(img).astype(float)
        depth_gt0 = (img / 255.0) * 15
        # depth at focal lenthg seem as 1
        depth_gtN = np.clip(depth_gt0, self.near, self.far)
        # log depth to strengthen detail in small depth
        depth_gtN = np.log10(depth_gtN)
        # normalize logged depth to 0 ~ 1 with
        depth_gtN = (depth_gtN - self.log_near) / (self.log_far - self.log_near)
        return depth_gtN, depth_gt0

    def get_pose(self, index):
        pose = self.poses[index]  # tX,tY,tZ,rX,rY,rZ,rW,time(s)
        pose = torch.from_numpy(pose).float()
        T = torch.eye(4).float()
        T[:3, :3] = tf3d.quaternion_to_matrix(pose[3:7])
        T[:3, 3] = pose[:3]
        return T


if __name__ == "__main__":
    ds = EndoSlamDataset("/data/Datasets/EndoSLAM/UnityCam/Colon", split='all', height=None, width=None)
    out = ds.get_depth(0)
    print()
