import cv2
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as tfm
from pathlib import Path
from geometry.camera import CameraModel, DepthBackprojD
import torchvision.transforms as tfm
from torchvision.datasets.folder import pil_loader
from tools.trainers.endodepth import plEndoDepth
from tools.visualize import visual_depth, visual_rgb
from tools.o3d_visual import np6d_o3d_color, o3d
import json


parser = argparse.ArgumentParser(
        description='Function to estimate depth maps of single or multiple images using an Endo-Depth model.')
parser.add_argument('--config', type=str, help='path to config json', required=True)

def predict(args):
    # Get log
    path = Path(args.log_path)
    ckpt = [f for f in (path / "checkpoints").glob("*.ckpt")]
    if len(ckpt) <= 0:
        raise RuntimeError("len(ckpt) <= 0")
    ckpt = str(ckpt[-1])
    hpam = path / "hparams.yaml"
    # Load model
    model = plEndoDepth.load_from_checkpoint(
        ckpt,
        test=True,
        hparams_file=str(hpam),
        map_location="cpu")

    # Load camera
    H, W = model.options.height, model.options.width
    cam = CameraModel(args.intrinsic_path)
    backproj = DepthBackprojD(H, W, cam.IK, cam.rd, cam.td)
    transform = tfm.Compose([
        tfm.ToTensor(),
        tfm.Resize((H, W)),
    ])

    images = Path(args.image_path)
    if images.is_dir():
        iter = tqdm(images.glob("*.png"))
    else:
        iter = tqdm([images])

    with torch.no_grad():
        for file in iter:
            # PREPROCESS
            color = pil_loader(file)
            color = transform(color)[None]
            depth = model.forward(color)

            # VISUALIZE
            pc = backproj.forward(depth)
            pc_col = torch.cat([pc[:, 0, :3], color], dim=1)
            pcd = np6d_o3d_color(pc_col.detach().cpu().view(6, -1).T.contiguous().numpy())

            if args.output_path is not None:
                outpath = Path(args.output_path).expanduser()
                if not outpath.exists():
                    import os
                    os.makedirs(outpath)

                f = file.stem
                src = visual_rgb(color, show=False)
                fname = '{}/{}.png'.format(outpath, f)
                cv2.imwrite(fname, src)
                img = visual_depth(depth, show=False)
                fname = '{}/{}.jpg'.format(outpath, f)
                cv2.imwrite(fname, img)
                fname = '{}/{}.npy'.format(outpath, f)
                np.save(fname, depth.cpu().numpy())
                fname = '{}/{}.ply'.format(outpath, f)
                o3d.io.write_point_cloud(fname, pcd, write_ascii=True)

            if args.show:
                o3d.visualization.draw_geometries([pcd], width=640, height=480)


if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)
    with open(args.config, 'r') as config_file:
        args_dict.update(json.load(config_file))
    predict(args)
