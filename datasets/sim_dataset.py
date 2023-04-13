from operator import inv
import os
import re
import numpy as np
from datasets.mono_dataset import MonoDataset
from geometry.camera import CameraModel


class SimDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(SimDataset, self).__init__(*args, **kwargs)

        self.cam = CameraModel("assets/blender.yaml")
        self.K = self.cam.K.float()
        self.invK = self.cam.IK.float()

        # self.near = 0.001  # ratio of focal length
        # self.far = 1000   # ratio of focal length
        self.near = 0.01  # ratio of focal length
        self.far = 100   # ratio of focal length
        self.focal = 20   # [mm]

        self.log_near = np.log10(self.near)
        self.log_far = np.log10(self.far)

        # Replace self.length for bidirection indexing
        self.load_dataset()
        self.length = len(self.filenames)

    def load_dataset(self):
        # Load image file names to id
        self.filenames = []
        folder = os.path.join(self.data_path, 'images')
        assert(os.path.exists(folder))
        if self.split == 'all':
            for file in os.listdir(folder):
                if file.endswith(self.img_ext):
                    self.filenames.append(file)
            import re
            self.filenames.sort(key=lambda f: int(re.sub(r'\D', '', f)))
        else:
            file = os.path.join(self.data_path, "{}.txt".format(self.split))
            assert(os.path.exists(file))
            with open(file, 'r') as f:
                self.filenames = f.readlines()

    def get_index(self, index):
        fname = self.filenames[index]
        match = re.match(r"(\d+)", fname)
        id = int(match[1])
        return id

    def get_color(self, index):
        file = "%04d.png" % index
        color = self.loader(os.path.join(self.data_path, 'images', file))
        return color

    def get_depth(self, index):
        file = "%08d_depth.npy" % index
        depth_path = os.path.join(self.data_path, "depth", file)
        depth_gt0 = np.load(depth_path) / self.focal  # !!!
        depth_gt0[depth_gt0 < 0] = self.far  # TODO: if this is meaningful?
        # depth at focal lenthg seem as 1
        depth_gtN = np.clip(depth_gt0, self.near, self.far)
        # log depth to strengthen detail in small depth
        depth_gtN = np.log10(depth_gtN)
        # normalize logged depth to 0 ~ 1 with
        depth_gtN = (depth_gtN - self.log_near) / (self.log_far - self.log_near)
        return depth_gtN, depth_gt0

    def get_pose(self, index):
        file = "extrinsic_%08d.txt" % index
        pose_path = os.path.join(self.data_path, "pose", file)
        pose = np.loadtxt(pose_path)
        if pose.shape != (4, 4):  # compliant with older dataset
            pose = np.vstack((pose, [0, 0, 0, 1]))
        pose[:3, 3] /= self.focal  # !!!
        return pose
    
    def get_intrinsic(self, index = None):
        return self.K, self.invK
