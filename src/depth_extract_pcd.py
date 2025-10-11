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
    print(f"Using checkpoint: {chkpt_path}")

    model = LInstantNGP.load_from_checkpoint(
        chkpt_path, map_location=COMPUTE_DEVICE, hparams_file=hparams_path
    )
    model.freeze()
    model.eval()

    data = U.lutils.NeRFData.load_from_checkpoint(
        chkpt_path, map_location=torch.device('cpu'), hparams_file=hparams_path
    )
    data._set_hparams(model.hparams)
    data.setup("predict")

    return model, data


def cloud_from_tensor(tens):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tens.cpu()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RayMeshNeRF Point Cloud generation script")
    parser.add_argument("log_name", help="Name of directory containing model under lightning_logs")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize final point cloud?")
    parser.add_argument("-a", "--angles", default=8, type=int, help="Count of angles to use for reconstruction")
    args = parser.parse_args()

    assert args.angles > 0  # Must be more than 0

    proj_dir = Path(f"{__file__}/../../").resolve()
    if not proj_dir.exists():
        raise ValueError(f"Specified logs at {proj_dir} don't exist")
    log_path = (proj_dir / "lightning_logs" / args.log_name).resolve()
    
    model, data = load_model_and_data(log_path)
    datadir = (proj_dir / "data" / data.hparams.source / data.hparams.name).resolve()

    idxs = U.data.find_mn_angles(data.c2ws, angle_count=args.angles)
    origin, direction = get_origin_direction_c2w_intrinsic(
        (data.images.shape[1], data.images.shape[2]),
        data.c2ws[idxs], data.intrinsics[idxs]
    )
    alpha_mask = data.images[idxs, ..., -1] != 0
    origin, direction = origin[alpha_mask], direction[alpha_mask]
    dl = DataLoader(TensorDataset(origin, direction), batch_size=2**9)

    print(f"Running Surface Point extraction for {dl.dataset.tensors[0].shape[0]:_d} rays")
    dm_depth= []
    with torch.no_grad():
        for o, di in tqdm(dl, total=len(dl), unit="batch", postfix="batch_size=2^9"):
            o, di = o.to(model.device), di.to(model.device)
            rgb, de, acc = model.render_rays(o, di)
            de = de.unsqueeze(-1)
            points = o + de * di
            mask = model.nerf(points, None, skip_colors=True) < model.hparams.f_sigma_threshold

            near_plane = torch.sqrt(torch.sum(torch.pow(o, 2), -1, keepdim=True)) + model.near_offset
            de[mask | (acc < 0.99) | (de < near_plane)] = torch.inf

            dm_depth.append(de.cpu())
        dm_depth = torch.cat(dm_depth, 0)

    #torch.save(dm_depth, "temp.pt")
    #dm_depth = torch.load("temp.pt")

    mask = (dm_depth != torch.inf).squeeze(-1)
    adjustment = (model.far_offset - model.near_offset) / 1024
    dm_points = origin[mask] + (dm_depth[mask] - adjustment) * direction[mask]
    bbox_mask = (dm_points.abs() <= 1.0).all(-1)
    dm_points = dm_points[bbox_mask]
    print(f"Points within [-1, 1] bbox limits: {dm_points.shape[0]:_d}")

    point_cloud = cloud_from_tensor(dm_points)
    cloud_path = datadir / f"dmn_cloud_raw{f'_{args.postfix}' if args.postfix else ""}.ply"
    o3d.io.write_point_cloud(cloud_path, point_cloud, write_ascii=True)
    print(f"Raw Point cloud written to {cloud_path}")

    exit(0)

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

    point_cloud = point_cloud.voxel_down_sample(0.002)
    point_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(100))
    print(f"After voxel downsample to 0.002 voxel size: {point_cloud}")

    print(f"Running DBScan clustering and Connectivity Merge filter...")
    labels = U.data.dbscan_and_connected_merge(point_cloud, eps=0.01, iters=3)
    v, c = np.unique_counts(labels)
    obj_label = v[c.argmax()]

    normals = np.asarray(point_cloud.normals)[labels == obj_label]
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(point_cloud.points)[labels == obj_label]))
    point_cloud.normals = o3d.utility.Vector3dVector(normals)
    point_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    print(f"After DBScan clustering and Connectivity Merge filter: {point_cloud}")

    if data.hparams.source == U.data.ObjectSource.DTU:
        point_cloud.transform(data.scaler)
    
    cloud_path = datadir / f"dmn_cloud{f'_{args.postfix}' if args.postfix else ""}.ply"
    o3d.io.write_point_cloud(cloud_path, point_cloud, write_ascii=True)
    print(f"Point cloud written to {cloud_path}")

    if args.visualize:
        o3d.visualization.draw_geometries([point_cloud])
