import torch
import numpy as np
import open3d as o3d
from PIL import Image as pil

from geometry.projective import DepthBackproj
from geometry.lie import se3_to_matrix, se3_to_matrix_inv


def make_point_cloud(image: torch.Tensor, depth: torch.Tensor, IK: torch.Tensor, fix_color=False):
    """ create a point cloud """

    _, _, H, W = image.shape
    colors = image.permute(0, 2, 3, 1)
    if fix_color:
        colors = colors[..., [2, 1, 0]] / 255.0

    iproj = DepthBackproj(H, W).to(image.device)
    # # K[:, 0, :] *= W
    # # K[:, 1, :] *= H
    # IK = torch.inverse(K)

    points = iproj.forward(depth, IK, norm=True).squeeze().view(4, -1)
    points = points[:3].permute(1, 0).view(-1, 3)
    pts = points.cpu().numpy()

    # open3d point cloud
    pc = o3d.geometry.PointCloud()

    clr = colors.squeeze().view(-1, 3).cpu().numpy()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(clr)

    return pc


def set_camera_pose(vis):
    """ set initial camera position """
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

    cam.extrinsic = np.array(
        [[0.91396544, 0.1462376, -0.37852575, 0.94374719],
         [-0.13923432, 0.98919177, 0.04597225, 1.01177687],
         [0.38115743, 0.01068673, 0.92444838, 3.35964868],
         [0., 0., 0., 1.]])

    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)


def sim3_visualization(G: torch.Tensor, images: torch.Tensor,
                       depths: torch.Tensor, Kn: torch.Tensor,
                       save=False, save_path="data/sim3", fix_color=False,
                       inv=False, verbose=0, show_frames=True, frame_size=1.0):
    """ convert depth to open3d point clouds 
        G: Transform point cloud from images[0] to images[1]
        IKn: Inverse of intrinsics that normalize with height and width.
        fix_color: If using uint8 color, then need to convert to float.
    """
    IK = torch.inverse(Kn.view(1, 4, 4)).view(4, 4)
    pc1 = make_point_cloud(images[:, 0], depths[:, :1], IK.clone(), fix_color=fix_color)
    pc2 = make_point_cloud(images[:, 1], depths[:, 1:], IK.clone(), fix_color=fix_color)
    frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)

    sim3_visualization.index = 1
    sim3_visualization.pc1 = pc1
    sim3_visualization.pc2 = pc2
    sim3_visualization.frame1 = frame1

    NUM_STEPS = 100
    dt = G / NUM_STEPS
    dT = se3_to_matrix(dt.view(1, 1, 6)) if not inv else se3_to_matrix_inv(dt.view(1, 1, 6))
    sim3_visualization.transform = dT.detach().cpu().view(4, 4).numpy()

    if verbose > 0:
        print(G)
        print(se3_to_matrix(G.view(1, 1, 6)))
    
    if show_frames:
        pass

    def animation_callback(vis):
        sim3_visualization.index += 1

        pc1 = sim3_visualization.pc1
        frame = sim3_visualization.frame1
        if sim3_visualization.index >= NUM_STEPS and \
                sim3_visualization.index < 2 * NUM_STEPS:

            pc1.transform(sim3_visualization.transform)
            frame.transform(sim3_visualization.transform)
            vis.update_geometry(pc1)
            vis.update_geometry(frame)
            vis.poll_events()
            vis.update_renderer()

            if save:
                image = vis.capture_screen_float_buffer()
                im = (np.asarray(image) * 255).astype(np.uint8)
                pil.fromarray(im).save(save_path + "/%06d.jpg" % sim3_visualization.index)

    vis = o3d.visualization.Visualizer()
    vis.register_animation_callback(animation_callback)
    vis.create_window(height=540, width=960)

    vis.add_geometry(pc1)
    vis.add_geometry(pc2)
    vis.add_geometry(frame1)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size))

    # vis.get_render_option().load_from_json("assets/renderoption.json")
    # set_camera_pose(vis)

    print("Press q to move to next example")
    vis.run()
    vis.destroy_window()
