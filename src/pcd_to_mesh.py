# Fix for draw_geometries crashing on Wayland
import os
os.environ["XDG_SESSION_TYPE"] = "x11"

import open3d as o3d
import argparse
from pathlib import Path
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="RayMeshNeRF Point Cloud to Mesh script, 'cloud' in filename swapped to 'mesh'"
    )
    parser.add_argument("pcd", help="Relative path to point cloud from project root directory, mesh saved to same dir")
    parser.add_argument("-d", "--depth", type=int, default=8, help="Poisson Surface Reconstruction Depth")
    parser.add_argument("-rq", "--remove_quantile", type=float, default=0.05,
                        help="Quantile limit of density to remove vertices below after PSR, prevents phantom planes")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize mesh?")
    parser.add_argument("-p", "--postfix", type=str, 
                        help="String to add after filename")
    args = parser.parse_args()

    proj_dir = Path(f"{__file__}/../../").resolve()
    if not proj_dir.exists():
        raise ValueError(f"Specified logs at {proj_dir} don't exist")
    pcd_path: Path = proj_dir / args.pcd

    pcd = o3d.io.read_point_cloud(pcd_path)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.depth)
    mesh.paint_uniform_color([0.5,0.5,0.5])
    mesh.compute_vertex_normals()

    if args.remove_quantile > 0.0:
        vertices_to_remove = densities < np.quantile(densities, args.remove_quantile)
        mesh.remove_vertices_by_mask(vertices_to_remove)

    # Remove unused vertices and tidy up
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()

    if args.visualize:
        o3d.visualization.draw_geometries([mesh])

    mesh_path = (pcd_path / ".." / f"{pcd_path.stem.replace("cloud", "mesh")}{f'_{args.postfix}' if args.postfix else ''}.ply").resolve()
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Mesh written to {mesh_path}")
