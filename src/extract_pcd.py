# Fix for draw_geometries crashing on Wayland
import os
os.environ["XDG_SESSION_TYPE"] = "x11"

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import argparse
from ingp import LInstantNGP
from pathlib import Path
import re
import open3d as o3d
import utils as U


COMPUTE_DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    COMPUTE_DEVICE = torch.device('cuda:0')
elif torch.mps.is_available():
    COMPUTE_DEVICE = torch.device('mps')
print(f"{COMPUTE_DEVICE=}")


def get_origin_direction_eq_angles(n, img_shape, focal):
    phis, thetas = U.rays.equidistance_rotations(n)

    origin, direction = [], []
    for phi, theta in zip(phis, thetas):
        o, d = U.rays.create_rays(
            img_shape[0],
            img_shape[1],
            U.rays.create_intrinsic(focal, img_shape),
            U.rays.look_at(4, phi, theta)
        )
        origin.append(o)
        direction.append(d)

    return torch.stack(origin), torch.stack(direction)


def get_origin_direction_c2w_intrinsic(img_shape, c2ws, intrinsics):
    origin, direction = [], []
    for c2w, intrinsic in zip(c2ws, intrinsics):
        o, d = U.rays.create_rays(img_shape[0], img_shape[1], intrinsic, c2w)
        origin.append(o)
        direction.append(d)

    return torch.stack(origin), torch.stack(direction)


def load_model_and_data(log_path):
    hparams_path = log_path / "hparams.yaml"

    chkpts = list((log_path / "checkpoints").glob("*best_val*"))
    chkpts.sort(key=lambda p: int(re.match(r".*epoch=(\d*).*", p.name, flags=re.DOTALL).group(1)))
    chkpt_path = chkpts[-1]

    model = LInstantNGP.load_from_checkpoint(
        chkpt_path, map_location=COMPUTE_DEVICE, hparams_file=hparams_path
    )
    model.freeze()
    model.eval()

    data = U.lutils.NeRFData.load_from_checkpoint(
        chkpt_path, map_location=torch.device('cpu'), hparams_file=hparams_path
    )
    data._set_hparams(model.hparams)
    data.setup("fit")

    return model, data


def cloud_from_tensor(tens):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tens.cpu()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RayMeshNeRF Point Cloud generation script")
    parser.add_argument("log_name", help="Name of directory containing model under lightning_logs")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize final point cloud?")
    parser.add_argument("-p", "--postfix", type=str, help="String to add after filename")
    args = parser.parse_args()

    proj_dir = Path(f"{__file__}/../../").resolve()
    if not proj_dir.exists():
        raise ValueError(f"Specified logs at {proj_dir} don't exist")
    log_path = (proj_dir / "lightning_logs" / args.log_name).resolve()
    
    model, data = load_model_and_data(log_path)

    idxs = U.data.find_rmn_angles(data.c2ws, angle_count=8)
    origin, direction = get_origin_direction_c2w_intrinsic((1200, 1600), data.c2ws[idxs], data.intrinsics[idxs])
    shape = origin.shape[:-1]
    dl = DataLoader(TensorDataset(origin.flatten(0, -2), direction.flatten(0, -2)), batch_size=2**19)

    print(f"Running RayMeshNeRF Surface Point extraction for {dl.dataset.tensors[0].shape[0]:_d} rays")
    rm_depth, sp_mask = [], []
    for o, d in tqdm(dl, total=len(dl), unit="batch", postfix="batch_size=2^19"):
        """
        d, sm = model.locate_density_gradient_based_surface_depth(
            o, d,
            sigma_limit=5.912066757, # sigma_limit=5.912066757,  # 0.01 / sqrt(3) * 1024
            gamma=1e-8,  # gamma=1e-9,
            non_grad_step_size=0.0016532793661392217, # non_grad_step_size=0.001691456,  # sqrt(3) / 1024
            min_step_size=1e-6,  # min_step_size=1 / 1024,  # 2 / (2 * sample count)
            max_iters=2**11 + 2**10,  # max_iters=2**12,  # 2*2048 as non_grad_step_size requires a min of 2048 steps from near to far
        )
        """
        d, sm = model.locate_density_gradient_based_surface_depth(
            o, d,
            sigma_limit=5.912066757, # sigma_limit=5.912066757,  # 0.01 / sqrt(3) * 1024
            gamma=1e-5,  # gamma=1e-9,
            non_grad_step_size=0.0016532793661392217, # non_grad_step_size=0.001691456,  # sqrt(3) / 1024
            min_step_size=1e-6,  # min_step_size=1 / 1024,  # 2 / (2 * sample count)
            max_iters=2**12,  # max_iters=2**12,  # 2*2048 as non_grad_step_size requires a min of 2048 steps from near to far
        )
        d, sm = d.detach().cpu(), sm.cpu()
        if model.device == torch.device("cuda:0"):
            torch.cuda.empty_cache()
        rm_depth.append(d)
        sp_mask.append(sm)

    rm_depth, sp_mask = torch.cat(rm_depth, 0).reshape(shape + (1,)), torch.cat(sp_mask, 0).reshape(shape)
    rm_depth_mask = ((rm_depth > model.hparams.near) & (rm_depth <= model.hparams.far)).squeeze(-1)
    rm_points = origin[rm_depth_mask] + rm_depth[rm_depth_mask] * direction[rm_depth_mask]

    print(f"Points within depth limits: {rm_points.shape[0]:_d}"
          f", of which {rm_points[sp_mask[rm_depth_mask]].shape[0]:_d} are Surface Points")

    bbox_mask = (rm_points.abs() <= 1.0).all(-1)
    point_cloud = cloud_from_tensor(rm_points[bbox_mask])
    print(f"After filtering bounding box [-1,1] outliers: {point_cloud}")

    point_cloud = point_cloud.voxel_down_sample(0.002)
    print(f"After voxel downsample to 0.002 voxel size: {point_cloud}")

    print(f"Running DBScan clustering and Connectivity Merge filter...")
    labels = U.data.dbscan_and_connected_merge(point_cloud, eps=0.01, iters=3)
    v, c = np.unique_counts(labels)
    obj_label = v[c.argmax()]
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(point_cloud.points)[labels == obj_label]))
    print(f"After DBScan clustering and Connectivity Merge filter: {point_cloud}")

    print(f"Calculating normal vectors...")
    normals = []
    normal_points = DataLoader(
        torch.from_numpy(np.asarray(point_cloud.points, dtype=np.float32)),
        batch_size=2**19, shuffle=False
    )
    for nps in normal_points:
        normals.append(model.estimate_normals(nps).cpu())
    normals = torch.cat(normals, dim=0)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)
    point_cloud.paint_uniform_color([0.5, 0.5, 0.5])

    datadir = (proj_dir / "data" / data.hparams.source / data.hparams.name).resolve()
    if data.hparams.source == U.data.ObjectSource.DTU:
        scaler = np.load(datadir / "cameras.npz")["scale_mat_0"]
        point_cloud.transform(scaler)
    
    cloud_path = datadir / f"rmn_cloud{f'_{args.postfix}' if args.postfix else ""}.ply"
    o3d.io.write_point_cloud(cloud_path, point_cloud, write_ascii=True)
    print(f"Point cloud written to {cloud_path}")

    if args.visualize:
        o3d.visualization.draw_geometries([point_cloud])
