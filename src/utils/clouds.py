import open3d as o3d
import torch
import numpy as np


def cloud_from_tensor(tens):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tens.cpu()))


def filter_cloud_by_mask(cloud: o3d.geometry.PointCloud, mask: np.ndarray):
    """Filter point cloud by points by copying, with colors and normals
    
    Args:
        cloud (N points): Original point cloud
        mask (shape[N]): Numpy mask for points

    Returns:
        cloud: Point cloud filtered by mask
    """
    data = {}
    data["points"] = np.asarray(cloud.points)[mask]
    if cloud.has_colors():
        data["colors"] = np.asarray(cloud.colors)[mask]
    if cloud.has_normals():
        data["normals"] = np.asarray(cloud.normals)[mask]

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["points"]))
    if "colors" in data:
        cloud.colors = o3d.utility.Vector3dVector(data["colors"])
    if "normals" in data:
        cloud.normals = o3d.utility.Vector3dVector(data["normals"])

    return cloud


def project_points_to_camera(points: torch.Tensor, c2w: torch.Tensor, intrinsic: torch.Tensor, 
                             image_size: tuple[int, int] = (800, 800)
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D points to camera coordinates and check visibility.
    
    Args:
        points (shape[N, 3]): Tensor of 3D points
        c2w (shape[4, 4]): Extrinisic camera matrix (Camera to World)
        intrinsic (shape[3, 3]): Intrinsic camera matrix
        image_size (height, width): Image dimensions
        
    Returns:
        uv (shape[N, 2)]): UV coordinates in image space
        depth (shape[N]): Depth values in world
        valid_mask (shape[N]): Boolean mask of points within image bounds
    """
    device = points.device
    N = points.shape[0]
    height, width = image_size

    # Defining depth in world space to make it more universal for Hash Encoding as it limits to a bbox here
    depth = torch.sqrt(torch.sum((points - c2w[:3, -1]) ** 2, dim=-1))
    
    # Utilizing projection to image space to find valid points
    points_homo = torch.cat([points, torch.ones(N, 1, device=device)], dim=1)
    w2c = torch.linalg.inv(c2w)
    points_cam = (w2c @ points_homo.T).T
    points_cam_3d = points_cam[:, :3]
    points_image = (intrinsic @ points_cam_3d.T).T
    
    # Points are valid if within image bounds and in front of cam
    uv = points_image[:, :2] / points_image[:, 2:3]
    u, v = uv[:, 0], uv[:, 1]
    valid_u = (u >= 0) & (u <= width)
    valid_v = (v >= 0) & (v <= height)
    # Using camera space depth is simpler here as world space depth doesn't have a sign
    valid_depth = points_cam[:, 2] < 0
    valid_mask = valid_u & valid_v & valid_depth
    
    return uv, depth, valid_mask


def calculate_point_visibility(points: torch.Tensor, normals: torch.Tensor, c2ws: torch.Tensor,
                               intrinsics: torch.Tensor, image_size: tuple[int, int] = (800, 800),
                               pixel_scaler: int = 1, depth_tolerance: float = 0.01) -> torch.Tensor:
    """
    Calculate visibility count for each point from multiple cameras.
    Points are visible if they are within depth_tolerance of the closest point for their pixel.
    
    Args:
        points (shape[N, 3]): Tensor of 3D points
        normals (shape[N, 3]): Normals for points
        c2ws (shape[M, 4, 4]): Extrinisic camera matrices (Camera to World)
        intrinsics (shape[M, 3, 3]): Intrinsic camera matrices
        image_size (height, width): Image dimensions
        pixel_scaler: pixels are considered as this size (1 is original, e.g. 2 makes width, height act like halved)
        depth_tolerance: points within this depth range of the closest point are considered visible
        
    Returns:
        visibility_count (shape[N]): Tensor with visibility count for each point
    """
    device = points.device
    N = points.shape[0]
    height, width = image_size
    
    visibility_count = torch.zeros(N, dtype=torch.int32, device=device)
    back_facing = torch.zeros(N, dtype=torch.int32, device=device)
    
    for c2w, intrinsic in zip(c2ws, intrinsics):
         # Usable for normal orientation
        to_cam_direction = torch.nn.functional.normalize(c2w[None, :3, -1] - points, p=2, dim=-1)
                                                         
        uv, depth, valid_mask = project_points_to_camera(points, c2w, intrinsic, image_size)
        
        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) == 0:
            continue
            
        valid_uv = uv[valid_indices]
        valid_depth = depth[valid_indices]
        
        # Finding closest pixel to point
        u_pixel = torch.round(valid_uv[:, 0]).long() // pixel_scaler * pixel_scaler
        v_pixel = torch.round(valid_uv[:, 1]).long() // pixel_scaler * pixel_scaler
        
        # Create pixel index tensor for grouping
        pixel_indices = v_pixel * width + u_pixel
        unique_pixels, inverse_indices = torch.unique(pixel_indices, return_inverse=True)
        
        # Closest points by depth per pixel
        min_depths = torch.zeros_like(unique_pixels, dtype=torch.float32)
        for i, pixel_idx in enumerate(unique_pixels):
            mask = (pixel_indices == pixel_idx)
            min_depths[i] = valid_depth[mask].min()
        
        # Keeping points within tolerance to depth
        point_min_depths = min_depths[inverse_indices]
        depth_differences = valid_depth - point_min_depths
        within_tolerance = depth_differences <= depth_tolerance
        
        # Mark visible points
        visible_indices = valid_indices[within_tolerance]
        visibility_count[visible_indices] += 1

        # If to_cam and normal faces opposite directions, we see the "back", usually an artifact of cloud generation
        dots = torch.sum(to_cam_direction[visible_indices] * normals[visible_indices], dim=-1)
        # Checking degenerate normals with the second term here
        back_facing[visible_indices] += (dots < 0) | (normals[visible_indices] == 0).all(-1)
        
    return visibility_count, back_facing


def get_visibility_mask(points: torch.Tensor, normals: torch.Tensor, c2ws: list[torch.Tensor],
                        intrinsics: list[torch.Tensor], image_size: tuple[int, int], pixel_scaler: int = 1,
                        depth_tolerance: float = 0.01, min_visibility: int = 1, max_back_face: int = 0
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Complete pipeline for processing NeRF point cloud with visibility filtering.
    
    Args:
        points (shape[N, 3]): Tensor of 3D points
        normals (shape[N, 3]): Normals for points
        c2ws (shape[M, 4, 4]): Extrinisic camera matrices (Camera to World)
        intrinsics (shape[M, 3, 3]): Intrinsic camera matrices
        image_size (height, width): Image dimensions
        pixel_scaler: pixels are considered as this size (1 is original, e.g. 2 makes width, height act like halved)
        depth_tolerance: points within this depth range of the closest point are considered visible
        min_visibility: minimum cameras that must see a point to keep it
        max_back_face: maximum allowed angles from which the point normal faces away from
        
    Returns:
        tuple: tuple containing (visibility_mask, back_facing_mask)
            - **visibility_mask**: *shape[N]*: Visibility mask of points above threshold
            - **back_facing_mask**: *shape[N]*: Back facing mask of points below threshold
    """
    visibility_count, back_facing = calculate_point_visibility(
        points, normals, c2ws, intrinsics, image_size, pixel_scaler, depth_tolerance
    )

    visibility_mask = visibility_count >= min_visibility
    back_facing_mask = back_facing > max_back_face
    return visibility_mask, back_facing_mask
