from torch.nn import Identity
from torch.utils.data.dataset import random_split
import torchvision.transforms as tfm

from datasets.sim_dataset import SimDataset
from tools.transforms import GammaJitter


def get_preprocess(height, width, istrain=True,
                   normalize=Identity()):
    if istrain:
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        gamma = (0.5, 2.0)
        transform = tfm.Compose([
            tfm.ToTensor(),
            tfm.Resize((height, width)),
            GammaJitter(gamma),
            tfm.ColorJitter(brightness, contrast, saturation, hue),
            normalize
        ])
        return transform

    else:
        transform = tfm.Compose([
            tfm.ToTensor(),
            tfm.Resize((height, width)),
            normalize
        ])
        return transform


def get_depth_dataset_only(data_path, height, width,
                           png: True, frame_ids=[0], istrain=True,
                           transforms=None):

    img_ext = '.png' if png else '.jpg'
    return SimDataset(
        data_path, "all", height, width,
        frame_idxs=frame_ids,
        img_ext=img_ext,
        use_random_idx=False,
        use_pose=False,
        use_depth=True,
        transforms=transforms or get_preprocess(height, width, istrain),
        load_mode='stack',
        bidirect=False,
    )


def get_depth_dataset(
        height, width, png: True,
        frame_ids=[0],
        data_path="/data/Datasets/blender/blender-duodenum-4-211027",
        val_path=None, split_ratio=0.95,
        transforms=None, transforms_val=None):

    img_ext = '.png' if png else '.jpg'

    if val_path is None:
        dataset = SimDataset(
            data_path, "all", height, width,
            frame_idxs=frame_ids,
            img_ext=img_ext,
            use_random_idx=False,
            use_pose=False,
            use_depth=True,
            transforms=transforms or get_preprocess(height, width, True),
            load_mode='stack',
            bidirect=False,
        )
        __L = len(dataset)
        __L1 = int(__L * split_ratio)
        __L2 = __L - __L1
        train_dataset, val_dataset = random_split(dataset, [__L1, __L2])

        from copy import deepcopy
        val_dataset.dataset = deepcopy(val_dataset.dataset)
        val_dataset.dataset.transforms = transforms_val or get_preprocess(height, width, False)

    else:
        dataset = SimDataset(
            data_path, "all", height, width,
            frame_idxs=frame_ids,
            img_ext=img_ext,
            use_random_idx=False,
            use_pose=False,
            use_depth=True,
            transforms=transforms or get_preprocess(height, width, True),
            load_mode='stack',
            bidirect=False,
        )
        train_dataset = dataset
        val_dataset = SimDataset(
            val_path, "all", height, width,
            frame_idxs=frame_ids,
            img_ext=img_ext,
            use_random_idx=False,
            use_pose=False,
            use_depth=True,
            transforms=transforms_val or get_preprocess(height, width, False),
            load_mode='stack',
            bidirect=False,
        )

    return train_dataset, val_dataset


def get_pose_dataset(
        height, width, png: True,
        frame_ids=[0],
        data_path="/data/Datasets/blender/blender-duodenum-4-211027",
        val_path=None, split_ratio=0.95):

    img_ext = '.png' if png else '.jpg'

    if val_path is None:
        dataset = SimDataset(
            data_path, "all", height, width,
            frame_idxs=frame_ids,
            img_ext=img_ext,
            use_random_idx=True,
            use_pose=True,
            use_depth=True,
            transforms=get_preprocess(height, width, True),
            load_mode='stack',
            search_region=[1, 60],
            bidirect=True,
        )
        __L = len(dataset)
        __L1 = int(__L * split_ratio)
        __L2 = __L - __L1
        train_dataset, val_dataset = random_split(dataset, [__L1, __L2])

        from copy import deepcopy
        val_dataset.dataset = deepcopy(val_dataset.dataset)
        val_dataset.dataset.transforms = get_preprocess(height, width, False)
    else:
        dataset = SimDataset(
            data_path, "all",
            frame_idxs=frame_ids,
            img_ext=img_ext,
            use_random_idx=True,
            use_pose=True,
            use_depth=True,
            transforms=get_preprocess(height, width, True),
            load_mode='stack',
            search_region=[1, 60],
            bidirect=True,
        )
        train_dataset = dataset
        val_dataset = SimDataset(
            val_path, "all", height, width,
            frame_idxs=frame_ids,
            img_ext=img_ext,
            use_random_idx=True,
            use_pose=True,
            use_depth=True,
            transforms=get_preprocess(height, width, False),
            load_mode='stack',
            search_region=[1, 10],
            bidirect=True,
        )

    return train_dataset, val_dataset
