import open3d as o3d
from pathlib import Path
import numpy as np
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import argparse
import torch

from utils.rays import create_rays
from utils.data import load_nesy_data


def get_origin_direction_c2w_intrinsic(img_shape, c2ws, intrinsics):
    origin, direction = [], []
    for c2w, intrinsic in zip(c2ws, intrinsics):
        o, d = create_rays(img_shape[0], img_shape[1], intrinsic, c2w)
        origin.append(o)
        direction.append(d)

    return torch.stack(origin), torch.stack(direction)


def get_ray_intersection_depths(mesh: o3d.geometry.TriangleMesh, origin: torch.Tensor, direction: torch.Tensor):
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh_t)

    rays = np.concat([origin.numpy(), direction.numpy()], axis=-1)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)
    return ans["t_hit"].numpy()


def sample_cloud_from_mesh(mesh: o3d.geometry.TriangleMesh, origin: torch.Tensor, direction: torch.Tensor,
                           samples: int = 2_500_000):
    chosen = torch.randperm(origin.shape[0])[:samples]
    origin, direction = origin[chosen], direction[chosen]

    depths = get_ray_intersection_depths(mesh, origin, direction)

    points = origin.numpy() + direction.numpy() * depths[..., None]
    points = points[~np.isinf(points).any(-1)]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return cloud


def mesh_to_cloud_signed_distances(mesh: o3d.geometry.TriangleMesh, cloud: o3d.geometry.PointCloud) -> np.ndarray:
    cloud = o3d.t.geometry.PointCloud.from_legacy(cloud)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    sdf = scene.compute_signed_distance(cloud.point.positions).abs()
    return sdf.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--postfix', type=str, default='')
    parser.add_argument('-s', '--scene', type=str, default='lego')
    parser.add_argument('-m', '--mode', type=str, default='mesh', choices=['mesh', 'pcd'])
    parser.add_argument('-r', '--runs', type=int, default=1)
    args = parser.parse_args()

    assert args.runs > 0

    root_dir = Path(f"{__file__}/../..").resolve()

    img, c2w, intrinsics, _ = load_nesy_data(args.scene, root_dir / "data", 1.0, "test")

    origin, direction = get_origin_direction_c2w_intrinsic(img.shape[1:3], c2w, intrinsics)
    alpha_mask = img[..., -1] > 0.9
    origin, direction = origin[alpha_mask], direction[alpha_mask]

    accuarcy, completeness = np.full((args.runs,), np.nan, np.float32), np.full((args.runs,), np.nan, np.float32)
    for i in range(args.runs):
        model = o3d.io.read_triangle_mesh(root_dir / f"data/NeSy/{args.scene}/model.ply")
        ground_truth = sample_cloud_from_mesh(model, origin, direction)

        if args.mode == 'pcd':
            reconstruction = o3d.io.read_point_cloud(
                root_dir / f"data/NeSy/{args.scene}/dmn_cloud{f'_{args.postfix}' if args.postfix else ''}.ply"
            )
        elif args.mode == 'mesh':
            mesh = o3d.io.read_triangle_mesh(
                root_dir / f"data/NeSy/{args.scene}/dmn_mesh{f'_{args.postfix}' if args.postfix else ''}.ply"
            )
            reconstruction = sample_cloud_from_mesh(mesh, origin, direction)

        accuarcy[i] = np.asarray(reconstruction.compute_point_cloud_distance(ground_truth)).mean()
        completeness[i] = np.asarray(ground_truth.compute_point_cloud_distance(reconstruction)).mean()

    accuarcy, completeness = accuarcy.mean().item(), completeness.mean().item()
    chamfer_distance = (accuarcy + completeness) / 2
    print(f"Accuracy: {accuarcy:8.6f} | Completeness: {completeness:8.6f} | Overall: {chamfer_distance:8.6f}")