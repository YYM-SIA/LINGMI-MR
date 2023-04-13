import numpy as np
from torchvision.datasets.folder import pil_loader
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as tfm


class Identity():  # used for skipping transforms
    def __call__(self, im):
        return im


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders
    """

    def __init__(self,
                 data_path, split,
                 height=None, width=None,
                 frame_idxs=[0],
                 img_ext='.jpg',
                 use_depth=True,
                 use_pose=True,
                 use_random_idx=False,
                 search_region=10,
                 transforms=None,
                 load_mode='split',
                 bidirect=True):
        super(MonoDataset, self).__init__()
        assert load_mode in ('split', 'stack')

        self.data_path = data_path
        self.split = split
        self.bidirect = bidirect

        # Implement these by yourself
        self.K = None
        self.invK = None

        self.frame_idxs = sorted(frame_idxs)
        self.use_random_idx = use_random_idx
        self.img_ext = img_ext
        self.load_depth = use_depth
        self.load_pose = use_pose
        self.filenames = None
        self.length = None
        self.load_mode = load_mode

        if isinstance(search_region, int):
            L = search_region
            assert L > 0
            self.search_region = [1, search_region]  # When use_random_idx
        elif isinstance(search_region, (list, tuple)):
            l = min(search_region)
            L = max(search_region)
            assert l > 0 and L > 0
            self.search_region = [l, L]  # When use_random_idx
        else:
            raise NotImplementedError()

        self.loader = pil_loader
        self.interp = tfm.InterpolationMode.BILINEAR

        resize = Identity()
        if height is not None and width is not None:
            resize = tfm.Resize((height, width), self.interp)

        self.transforms = transforms  # TODO:
        if self.transforms is None:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            self.transforms = tfm.Compose([
                resize,
                tfm.ToTensor(),
                tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.transformsD = tfm.Compose([
            tfm.ToTensor(),
            resize
        ])

    def __len__(self):
        scale = 1 if not self.bidirect else 2
        if self.length is None:
            assert self.filenames is not None, "Please load your dataset to self.filenames firstly."
            return len(self.filenames) * scale
        else:
            return self.length * scale

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            "pose"                                  for camera pose,
            "ipose"                                 for inverse camera pose,
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        idx = [
            index + i * (np.random.randint(*self.search_region) if self.use_random_idx else 1)
            for i in self.frame_idxs
        ]

        # Load data
        inputs['index'] = []
        inputs['color'] = []
        if self.load_depth:
            inputs['depth_norm'] = []
            inputs['depth'] = []
        if self.load_pose:
            inputs['pose'] = []

        for __id, i in zip(idx, self.frame_idxs):
            __index = self.parse_index(__id)
            inputs['index'].append(__index)

            __image = self.get_color(__index)
            __image = self.transforms(__image)
            inputs['color'].append(__image)

            if self.load_depth:
                __depthN, __depth0 = self.get_depth(__index)
                __depthN = np.expand_dims(__depthN, 2)
                __depth0 = np.expand_dims(__depth0, 2)
                __depthN = self.transformsD(__depthN.astype(np.float32))
                __depth0 = self.transformsD(__depth0.astype(np.float32))
                inputs['depth_norm'].append(__depthN)
                inputs['depth'].append(__depth0)

            if self.load_pose:
                __pose = self.get_pose(__index)
                if not isinstance(__pose, torch.Tensor):
                    __pose = torch.from_numpy(__pose.astype(np.float32))
                inputs['pose'].append(__pose)

        # Stack data
        if self.load_mode == 'stack':
            inputs['color'] = torch.stack(inputs['color'], dim=0)
            if self.load_depth:
                inputs['depth_norm'] = torch.stack(inputs['depth_norm'], dim=0)
                inputs['depth'] = torch.stack(inputs['depth'], dim=0)
            if self.load_pose:
                inputs['pose'] = torch.stack(inputs['pose'], dim=0)
        return inputs

    def parse_index(self, index):
        index = np.clip(index, 0, self.__len__() - 1)
        if self.bidirect:
            # Bidirection indexing to make data continous
            index = index % (self.length * 2)
            if index >= self.length:
                index = (2 * self.length - 1) - index
        else:
            index = index % self.length
        return self.get_index(index)

    def get_index(self, index):
        """ Get the index to load target file.
        """
        raise NotImplementedError

    def get_color(self, index):
        raise NotImplementedError

    def get_depth(self, index):
        raise NotImplementedError

    def get_pose(self, index):
        raise NotImplementedError

    def get_intrinsic(self, index = None):
        raise NotImplementedError
