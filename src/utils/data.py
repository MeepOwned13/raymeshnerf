import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from enum import StrEnum
from PIL import Image
import cv2
from copy import copy
import open3d as o3d
import matplotlib.pyplot as plt

from .rays import create_rays, create_intrinsic, equidistance_rotations, look_at
from .mesh_render import render_gso_mesh


def create_nerf_data(images: Tensor, c2ws: Tensor, intrinsics: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Creates rays for NeRF training

    Args:
        image (shape[N, H, W, 3-4]): Images to extract colors and sizes from
        c2w (shape[N, 4, 4]): Extrinisic camera matrices (Camera to World)
        intrinsic (shape[N, 3, 3]): Intrinsic camera matrices

    Returns:
        tuple: tuple containing (origins, directions, colors, pixel_weights)
            - **origins**: *shape[N * H * W, 3]*: Ray origins in World coordinates
            - **directions**: *shape[N * H * W, 3]*: Cartesian ray directions in World
            - **colors**: *shape[N * H * W, 3-4]*: RGB(A) colors for rays
    """
    origins, directions, colors = [], [], []

    # Collecting to list then concat for ease
    for image, c2w, intrinsic in zip(images, c2ws, intrinsics):
        o, d = create_rays(image.shape[0], image.shape[1], intrinsic, c2w)

        origins.append(o.flatten(0, 1))
        directions.append(d.flatten(0, 1))
        colors.append(image.flatten(0, 1))

    origins = torch.cat(origins, dim=0)
    directions = torch.cat(directions, dim=0)
    colors = torch.cat(colors, dim=0)

    return origins, directions, colors


def find_val_angles(c2ws: torch.Tensor, angle_count: int = 12) -> torch.Tensor:
    """Deterministically get validation angle indicies from extrinsic camera matrices

    Takes angle_count many angles corresponding to equidistant points on the unit sphere, and finds closest cameras
    (normalized to unit sphere) from them to get validation points, covering the scene evenly.

    Args:
        c2ws (shape[N, 4, 4]): Extrinisic camera matrices (Camera to World)
        angle_count: How many angles to choose

    Returns:
        idxs (shape[angle_count]): Indicies of chosen validation angles
    """
    cam_pos = c2ws[:, :3, -1].clone()
    cam_pos = torch.nn.functional.normalize(cam_pos, "fro", -1)

    part_pos = torch.zeros((angle_count, 3), dtype=torch.float32)
    for i, (phi, theta) in enumerate(zip(*equidistance_rotations(angle_count))):
        part_pos[i] = look_at(1, phi, theta)[:3, -1]

    # N,1,3 | 1,K,3 -> N,K
    distances = torch.sqrt(torch.sum(torch.pow(cam_pos.unsqueeze(1) - part_pos.unsqueeze(0), 2), dim=-1))

    return torch.argmin(distances, dim=0)


def find_rmn_angles(c2ws: torch.Tensor, angle_count: int = 8) -> list[int]:
    """Deterministically get reconstruction angle indicies from extrinsic camera matrices using FPS

    Utilizes Farthest Point Sampling for angle choices, first angle is chosen as the farthest camera position from
    the centroid of camera points (mean of coordinates).

    Args:
        c2ws (shape[N, 4, 4]): Extrinisic camera matrices (Camera to World)
        angle_count: How many angles to choose

    Returns:
        idxs (len=angle_count): Indicies of chosen reconstruction angles
    """
    cam_pos = c2ws[:, :3, -1].clone()
    picked = [torch.sqrt(torch.sum(torch.pow(cam_pos - cam_pos.mean(), 2), -1)).argmax().item()]
    distances = torch.full((cam_pos.shape[0],), torch.inf, dtype=torch.float32)

    for _ in range(1, angle_count):
        last = cam_pos[picked[-1]]
        distance_to_last = torch.sqrt(torch.sum(torch.pow(cam_pos - last, 2), -1))
        distances = torch.minimum(distances, distance_to_last)
        picked.append(torch.argmax(distances).item())

    return picked


def compute_near_far_planes(c2ws: Tensor) -> tuple[float, float]:
    """Compute minimal near and maximal far plane for ray sampling

    Computes minimal and maximal distance to [-1, 1] bbox corners by first computing distance to origins based on
    camera to world transformations and then adds/removes sqrt(3) from it (min-max distance to bbox corner from center)

    Args:
        c2ws (shape[K, 4, 4]): Extrinsic camera matrices (Camera to World)

    Returns:
        tuple: tuple containing (near_plane, far_plane)
        - **near_plane**: Minimal near plane found
        - **far_plane**: Maximal far plane found
    """
    distance_from_center = c2ws[:, :3, -1].norm(dim=-1)
    max_dfc = distance_from_center.max()
    min_dfc = distance_from_center.min()

    # Distance to box corner is maximal at sqrt(3) for [-1, 1] bbox
    near = max(0.0, (min_dfc - torch.sqrt(torch.tensor(3))).item())
    far = (max_dfc + torch.sqrt(torch.tensor(3))).item()

    return near, far


def dbscan_and_connected_merge(cloud: o3d.geometry.PointCloud, eps: float = 0.01, merge_distance: float = 0.005,
                               iters: int = 5, color_cloud: bool = False) -> np.ndarray:
    """Apply DBScan clustering and merge connected clusters (by minimal distance between them)

    This algorithm is non deterministic in selecting the next cluster for merging, so multiple iterations are required.
    The runtime can explode when looking at large clouds so make sure to downsample if the function doesn't run in a
    reasonable time.

    Args:
        cloud: PointCloud to apply dbscan and connection merging to
        eps: Epsilon for dbscan (o3d.geometry.PointCloud.cluster_dbscan)
        merge_distance: Max distance when merging of clusters still happens
        iters: How many iterations to run, set it higher if connected close areas still get separately labeled
        color_cloud: Apply a coloring based on labels to cloud?
    
    Returns:
        labels: Label per point for the original cloud
    """
    labels = np.array(cloud.cluster_dbscan(eps=eps, min_points=1))
    relabels = labels[labels != -1]  # DBScan identified noise
    # Everything "plays" that merged things into itself
    playing = set(np.unique(labels).tolist())
    for _ in range(iters):
        new_playing = set()

        while len(playing) != 0:
            dst = playing.pop()
            dst_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(cloud.points)[relabels == dst]))

            for src in copy(playing):  # Copy to avoid changing iterator while iterating
                src_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(cloud.points)[relabels == src]))
                mindist = np.asarray(dst_cloud.compute_point_cloud_distance(src_cloud)).min()

                if mindist <= merge_distance:
                    playing.remove(src)
                    dst_cloud = dst_cloud + src_cloud
                    relabels[relabels == src] = dst

            # Played if it's close to anything, if it isn't then it'll never be
            new_playing.add(dst)

        playing = new_playing

    _, relabels = np.unique(relabels, return_inverse=True)
    labels[labels != -1] = relabels  # Keep identified noise labels

    if color_cloud:
        colors = plt.get_cmap("tab20")(labels / (labels.max() if labels.max() > 0 else 1))
        colors[labels < 0] = 0
        cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return labels


class ObjectSource(StrEnum):
    GSO = "GSO"
    DTU = "DTU"
    NeSy = "NeSy"


def load_gso_data(name: str, directory: str, sensor_count: int = 64, size: int = 800):
    obj_dir: Path = (Path(directory) / ObjectSource.GSO.value / name).resolve()
    npz_path: Path = obj_dir / "render.npz"

    if not npz_path.exists():
        if not obj_dir.is_dir():
            raise ValueError(f"[GSO] Directory of object '{name}' doesn't exist")

        images, c2ws, focal = render_gso_mesh(
            obj_path=obj_dir,
            sensor_count=sensor_count,
            size=size,
        )
        np.savez_compressed(npz_path, images=images, c2ws=c2ws, focal=focal)

    data = np.load(npz_path)

    images = torch.from_numpy(data["images"]).to(torch.float32)
    c2ws = torch.from_numpy(data["c2ws"]).to(torch.float32)
    focal = torch.from_numpy(data["focal"]).to(torch.float32)
    intrinsics = create_intrinsic((focal, focal), (size, size)).unsqueeze(0).expand(c2ws.shape[0], -1, -1)

    return images, c2ws, intrinsics


# This function is borrowed and modified from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K, R, t = out[0], out[1], out[2]
    intrinsic = np.astype(K / K[2, 2], np.float32)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    # transforming z-forward, y-down to z-backward, y-up
    pose[:3, 1] *= -1
    pose[:3, 2] *= -1
    return pose, intrinsic


def load_dtu_data(name: str, directory: str, masked: bool = True):
    scan_dir: Path = (Path(directory) / ObjectSource.DTU.value / name).resolve()

    f_images = (scan_dir / "image").glob("[0-9]*.png")
    f_masks = (scan_dir / "mask").glob("[0-9]*.png")
    cameras = np.load(scan_dir / "cameras.npz")

    images, c2ws, intrinsics = [], [], []
    for i, (f_img, f_mask) in enumerate(zip(sorted(f_images), sorted(f_masks))):
        image = np.asarray(Image.open(f_img), dtype=np.float32) / 255.0
        if masked:
            alpha = np.asarray(Image.open(f_mask), dtype=np.float32) / 255.0
            image = np.concat([image, alpha.mean(-1, keepdims=True)], axis=-1)

        proj_matrix = cameras[f'world_mat_{i}'] @ cameras[f'scale_mat_{i}']
        proj_matrix = proj_matrix[:3, :4]
        c2w, intrinsic = load_K_Rt_from_P(proj_matrix)
        
        images.append(torch.from_numpy(image))
        c2ws.append(torch.from_numpy(c2w))
        intrinsics.append(torch.from_numpy(intrinsic))

    images = torch.stack(images, dim=0)
    c2ws = torch.stack(c2ws, dim=0)
    intrinsics = torch.stack(intrinsics, dim=0)

    return images, c2ws, intrinsics


def load_data(name: str, source: ObjectSource = ObjectSource.GSO, directory: str | None = None,
              **kwargs) -> tuple[Tensor, Tensor, Tensor]:
    """Loads object data from disk, or renders if doesn't exist, follows Google Scanned Objects mesh format

    Args:
        name: Name of object directory under directory
        source: Source of data (dataset idenfitifer based on enum)
        directory: Directory to search objects under, defaults to project_root/data
        **kwargs: Refer to load_SOURCE_data function arguments

    Returns:
        tuple: tuple containing (images, c2ws, focal)
        - **images**: *shape[N, H, W, 3]*: Images
        - **c2ws**: *shape[N, 4, 4]*: Extrinisic camera matrices (Camera to World)
        - **intrinsics**: *shape[N, 3, 3]*: Intrinsic camera matrices
    """
    directory = directory or f"{__file__}/../../../data"

    match source:
        case ObjectSource.GSO:
            data = load_gso_data(name, directory, **kwargs)
        case ObjectSource.DTU:
            data = load_dtu_data(name, directory, **kwargs)
        case ObjectSource.NeSy:
            raise NotImplementedError(source)
        case _:
            raise ValueError(f"Unknown ObjectSource: {source}")

    return data
