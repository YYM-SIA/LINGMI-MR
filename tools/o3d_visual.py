import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def npxyz_o3d(xyz: np.ndarray):
    """
    Convert Nx3 numpy array to open3d point cloud. 
    Args:
        xyz (np.ndarray): Nx3 numpy array points
    Returns:
        [type]: [description]
    """
    assert(len(xyz.shape) == 2)
    assert(xyz.shape[1] == 3)
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def o3d_npxyz(pcd: o3d.geometry.PointCloud):
    return np.asarray(pcd.points)


def npxyzw_o3d(xyzw: np.ndarray, cmap='viridis'):
    """
    Convert Nx4 numpy array to open3d point cloud. 
    Args:
        xyzw (np.ndarray):Nx4 numpy array points with value (eg. TSDF)
        cmap (str, optional): matplotlib color maps. Defaults to ''.
    Returns:
        [type]: [description]
    """
    assert(len(xyzw.shape) == 2)
    assert(xyzw.shape[1] == 4)
    cmap = plt.cm.get_cmap(cmap)
    rgb = cmap(xyzw[:, 3])[:, :3]
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzw[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def np6d_o3d_normal(pose: np.ndarray):
    """
    Convert Nx6 numpy array to open3d point cloud with normals. 
    Args:
        pose (np.ndarray):Nx6 numpy array points with normals.
    Returns:
        [type]: [description]
    """
    assert(len(pose.shape) == 2)
    assert(pose.shape[1] == 6)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pose[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(pose[:, 3:])
    return pcd


def np6d_o3d_color(point: np.ndarray):
    """
    Convert Nx6 numpy array to open3d point cloud with colors. 
    Args:
        pose (np.ndarray):Nx6 numpy array points with colors.
    Returns:
        [type]: [description]
    """
    assert(len(point.shape) == 2)
    assert(point.shape[1] == 6)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point[:, 3:])
    return pcd


def o3d_show(items, *args, **kwargs):
    o3d.visualization.draw_geometries(items, *args, **kwargs)
