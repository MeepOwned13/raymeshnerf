import torch
from torch import Tensor, nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


# In accordance with mitsuba's conventions
def look_at(radius: float, phi: Tensor, theta: Tensor) -> Tensor:
    """Construct Look At matrix for [0, 0, 0] target

    Args:
        radius: Distance to target, aka. radius of sphere the camera is laying on
        phi (shape[]): Horizontal rotation in radians
        theta (shape[]): Vertical rotation in radians

    Returns:
        look_at shape([4, 4]): Look at matrix in homogeneous coordinates
    """
    origin = torch.tensor([
        radius * torch.sin(theta) * torch.cos(phi),
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.cos(theta),
    ])
    
    target = torch.tensor([0,0,0], dtype=torch.float32)
    forward = F.normalize(origin - target, p="fro", dim=0)
    up = torch.tensor([0, 0, 1], dtype=torch.float32)
    right = F.normalize(torch.cross(up, forward, dim=0), p="fro", dim=0)
    up = F.normalize(torch.cross(forward, right, dim=0), p="fro", dim=0)

    return torch.tensor([
        [right[0], up[0], forward[0], origin[0]],
        [right[1], up[1], forward[1], origin[1]],
        [right[2], up[2], forward[2], origin[2]],
        [0, 0, 0, 1],
    ])


def equidistance_rotations(n: int) -> tuple[Tensor, Tensor]:
    """Calculate equidistance rotations along sphere
    
    Args:
        n: Angle count
    
    Returns:
        tuple: tuple containing(phis, thetas)
            - **phis**: *shape[n]*: Phis - horizontal rotations - in radians
            - **thetas**: *shape[n]*: Thetas - vertical rotations - in radians
    """
    i = torch.arange(0, n, dtype=torch.float32) + 0.5
    phis = torch.pi * i * (1 + torch.sqrt(torch.tensor(5))) % (2 * torch.pi)
    thetas = torch.arccos(1 - 2 * i / n)

    return phis, thetas


def create_intrinsic(focal: Tensor | tuple | list, size: Tensor | tuple | list):
    """Create intrinsic matrix from focal length and image size

    Args:
        focal (shape[2]): Focal length in pixels for x and y
        size (shape[2]): Image width and height

    Returns:
        intrinsic (shape[3, 3]): Intrinsic camera matrix in homogeneous coordinates
    """
    return torch.tensor([
        [focal[0], 0, size[0] // 2],
        [0, focal[1], size[1] // 2],
        [0, 0, 1],
    ], dtype=torch.float32)


def create_rays(height: int, width: int, intrinsic: Tensor, c2w: Tensor) -> tuple[Tensor, Tensor]:
    """Create rays cast by camera in World coordinates, assumes X-right, Y-up, Z-backward system

    Args:
        height: Rays to take vertically (image height)
        width: Rays to take horizontally (image width)
        intrinsic (shape[3, 3]): Intrinsic camera matrix
        c2w (shape[4, 4]): Extrinsic camera matrix (Camera to World)

    Returns:
        tuple: tuple containing (ray_origins, ray_directions)
        - **ray_origins**: *shape[width, height, 3]*: Ray origins in World coordinates
        - **ray_directions**: *shape[width, height, 3]*: Cartesian ray directions in World
    """
    device = c2w.device

    # Doing everything on cpu so later ops don't run out of 2^19 kernel
    c2w, intrinsic = c2w.cpu(), intrinsic.cpu()
    focal_x = intrinsic[0, 0]
    focal_y = intrinsic[1, 1]
    # cx and cy handle the misalignement of the principal point with the center of the image
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # Index each point on the image, determine ray directions to them
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing='xy'
    )
    directions = torch.stack((
        (i - cx) / focal_x,
        -(j - cy) / focal_y,
        -torch.ones(i.shape, dtype=torch.float32)  # -1 since ray is cast away from camera
    ), -1)

    # Transform ray directions to World, origins just need to be broadcasted accordingly
    ray_directions = F.normalize(directions @ c2w[:3, :3].T, "fro", -1)
    ray_origins = torch.broadcast_to(c2w[:3, -1], ray_directions.shape)  # c2w last column determines position

    return ray_origins.to(device), ray_directions.to(device)


def sample_ray_uniformally(origins: Tensor, directions: Tensor, near_offset: float, far_offset: float,
                           num_samples: int, perturb=True) -> tuple[Tensor, Tensor, Tensor]:
    """Uniformally sample rays and return them in the World coordinate system

    Near and far are defined as offsets, depth are gained by adding [near, far] to the distance from origins to (0,0,0),
    the setup accomodates situations where scenes are contained within spheres/cubes where radius is easily defined. In
    such cases near=-radius, far=radius.

    Args:
        origins (shape[N, 3]): Ray origins in World coordinates
        directions (shape[N, 3]): Cartesian ray directions in World
        near_offset: Near plane offset from World origin, generally negative
        far_offset: Far plane offset from World origin, generally positive
        num_samples: How many samples to take along the ray
        perturb: If True, adds noise to the depths

    Returns:
        tuple: tuple containing (points, directions, depths)
        - **points**: *shape[N, num_samples, 3]*: Sampled points in World coordinates
        - **directions**: *shape[N, num_samples, 3]*: Original directions expanded to match the shape of points
        - **depths**: *shape[N, num_samples]*: Depth of each sampled point on the given ray
    """
    device = origins.device
    offsets = torch.linspace(near_offset, far_offset, num_samples, dtype=torch.float32, device=device).unsqueeze(0)
    dist_to_zero = torch.sqrt(torch.sum(torch.pow(origins, 2), -1, keepdim=True))
    depths = (dist_to_zero + offsets)

    if perturb:
        # Noise is at most half of step size, this ensures sorted depths, required for volume rendering
        noise = (torch.rand(depths.shape, device=device) - 0.5) * (far_offset - near_offset) / num_samples / 2
        depths = (depths + noise)

    points = origins[..., None, :] + directions[..., None, :] * depths[..., :, None]
    # Expand directions to make NeRF input
    directions = directions[..., None, :].expand(points.shape)
    return points, directions, depths


def sample_pdf(bins: Tensor, weights: Tensor, num_samples: int, deterministic: bool = False) -> Tensor:
    """Samples based on an approximated Probability Density Function

    Args:
        bins (shape[N, M]): Bin bounds
        weights (shape[N, M]): Weights of bins
        num_samples: How many samples to take
        deterministic: If True, uses a linspace (also ensures sorted output) to re-sample instead of random

    Returns:
        samples(shape[N, num_samples]): Set of new samples based on approximated PDF
    """
    device = weights.device

    weights = weights + 1e-5  # avoid nans later
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # Normalize PDF
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=device), cdf], dim=-1)  # Prepend 0 to have cdf->[0,1]

    if deterministic:
        u = torch.linspace(0.0, 1.0, steps=num_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=device)

    # Inverting the CDF
    u = u.contiguous()  # Need contigous memory layout for further operations
    indexes = torch.searchsorted(cdf, u, right=True)  # Finding bins
    # Need to ensure below and above don't leave bounds of bins
    below = torch.max(torch.zeros_like(indexes - 1), indexes - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(indexes), indexes)
    indexes = torch.stack([below, above], dim=-1)

    # Gathering sampled bins and bound probabilites
    shape = [indexes.shape[0], indexes.shape[1], cdf.shape[-1]]
    cdf = torch.gather(cdf.unsqueeze(1).expand(shape), dim=2, index=indexes)
    bins = torch.gather(bins.unsqueeze(1).expand(shape), dim=2, index=indexes)

    # denominator is the size of the bins
    denominator = cdf[..., 1] - cdf[..., 0]
    denominator = torch.where(denominator < 1e-5, torch.ones_like(denominator), denominator)
    denominator[denominator < 1e-5] = 1.0
    # t gives the relative position inside the bins
    t = (u - cdf[..., 0]) / denominator

    samples = bins[..., 0] + t * (bins[..., 1] - bins[..., 0])
    return samples


def sample_ray_hierarchically(origins: Tensor, directions: Tensor, num_samples: int, bins: Tensor,
                              weights: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    """Hierarchically sample rays and return them in the World coordinate system

    Args:
        origins (shape[N, 3]): Ray origins in World coordinates
        directions (shape[N, 3]): Cartesian ray directions in World
        num_samples: How many samples to take along the ray
        bins (shape[N, M]): Bin bounds calculated from a previous sampling
        weights (shape[N, M]): Weights of bins calculated frm a previous sampling
        deterministic: If True, uses a linspace (also ensures sorted output) to re-sample instead of random


    Returns:
        tuple: tuple containing (points, directions, depths)
        - **points**: *shape[N, num_samples, 3]*: Sampled points in World coordinates
        - **directions**: *shape[N, num_samples, 3]*: Original directions expanded to match the shape of points
        - **depths**: *shape[N, num_samples]*: Depth of each sampled point on the given ray
    """
    depths = sample_pdf(bins, weights, num_samples, deterministic=deterministic)

    points = origins[..., None, :] + directions[..., None, :] * depths[..., :, None]
    # Expand directions to make NeRF input
    directions = directions[..., None, :].expand(points.shape)
    return points, directions, depths


def plot_ray_sampling(points: Tensor, origin: Tensor, cartesian_direction: Tensor, title: str):
    """Create a 3D plot of rays from the implied camera's view and a rotated onex

    Args:
        points (shpe[N, num_samples, 3]): Sampled points in World coordinates
        origin (shape[3]): Origin of all rays in World coordinates
        cartesian_direction([3]): View direction for the first subplot
        title: Title of the plot
    """
    points = points.cpu()
    origin = origin.cpu()
    cartesian_direction = cartesian_direction.cpu()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={"projection": "3d"})
    axes = axes.flatten()
    fig.suptitle(title)
    plt.tight_layout()
    # Adding the origin so it always starts from the camera position
    points = torch.cat([origin.expand((points.shape[0], 1, -1)), points], 1)

    # Convert to spherical coordinates
    X, Y, Z = -cartesian_direction  # Taking the negative as view_init specifies direction outward
    R = torch.sqrt(X**2 + Y**2 + Z**2)
    X, Y, Z = X / R, Y / R, Z / R  # normalization
    azim = torch.rad2deg(torch.atan2(Y, X))
    elev = torch.rad2deg(torch.arcsin(Z))
    # Multiple angles to understand better
    for ax, (mod_elev, mod_azim) in zip(axes, [[0, 0], [-10, 60]]):
        ax.view_init(elev + mod_elev, azim + mod_azim, 0)
        ax.plot(points[:, :, 0], points[:, :, 1], points[:, :, 2], linewidth=0.2, markersize=2, marker='o')
    plt.show()


def depths_to_distance(origins: Tensor, depths: Tensor, far_offset: float):
    """Get distances between sample points from depths and far_offset, uses World coordinates

    Last distance is calculated from far plane (or 0 if point is beyond it)

    Args:
        origins (shape[N, 3]): Ray origins in World coordinates
        depths (shape[N, M]): Specifies how far along the rays are the RGBSs
        far_offset: Far plane offset from World origin, generally positive
    """
    dist_to_zero = torch.sqrt(torch.sum(torch.pow(origins, 2), -1, keepdim=True))
    far_planes = (dist_to_zero + far_offset)
    distances = depths[..., 1:] - depths[..., :-1]
    distances = torch.cat([distances, F.relu(far_planes - depths[..., -1:])], -1)
    return distances


def render_rays(origins: Tensor, rgbs: Tensor, depths: Tensor, far_offset: float) -> tuple[Tensor, Tensor, Tensor]:
    """Performs Volumetric Rendering

    Args:
        origins (shape[N, 3]): Ray origins in World coordinates
        rgbs (shape[N, M, 4]): RGB and Sigma values for sampled points
        depths (shape[N, M]): Specifies how far along the rays are the RGBSs
        far_offset: Far plane offset from World origin, generally positive

    Returns:
        tuple: a tuple containing (rgb, depth, acc) where
        - **rgb**: *shape[N, 3]*: RGB value calculated for ray,
        - **depth**: *shape[N]*: Approximated depth of ray termination,
        - **acc**: *shape[N, 1]*: Sum of weights for pixel (alpha)
        - **weights**: *shape[N, M]*: Render weight per sample point
    """
    device = rgbs.device
    distances = depths_to_distance(origins, depths, far_offset)

    alpha = 1.0 - torch.exp(-F.relu(rgbs[..., 3]) * distances)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + torch.finfo(rgbs.dtype).eps], -1), -1
    )[:, :-1]

    rgb = torch.sum(weights[..., None] * rgbs[..., :3], dim=-2)
    depth = torch.sum(weights * depths, dim=-1)
    acc = torch.sum(weights, dim=-1).unsqueeze(-1).clamp(0.0, 1.0)  # Clamp to counter numerical errors

    return rgb, depth, acc, weights


def get_render_weights(origins: Tensor, sigma: Tensor, depths: Tensor, far_offset: float,
                       re_weigh_alpha: float = 1.0) -> Tensor:
    """Calculates weights for Volumetric Rendering

    Args:
        origins (shape[N, 3]): Ray origins in World coordinates
        sigma (shape[N, M, 1]): Sigma values for sampled points
        depths (shape[N, M]): Specifies how far along the rays are the RGBSs
        far_offset: Far plane offset from World origin, generally positive
        re_weigh_alpha: Allows for making alpha 1.0, where it wouldn't be, rendering thin surfaces fully, any ray where
            the alpha is above this parameter gets scaled

    Returns:
        weights (shape[N, M]): Render weight per sample point
    """
    device = sigma.device
    distances = depths_to_distance(origins, depths, far_offset)

    alpha = 1.0 - torch.exp(-F.relu(sigma.squeeze(-1)) * distances)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + torch.finfo(sigma.dtype).eps], -1), -1
    )[:, :-1]

    if re_weigh_alpha < 1.0:
        alpha = weights.sum(-1, keepdim=True).clamp(0, 1.0)
        weights = weights / torch.where(alpha > re_weigh_alpha, alpha, 1.0)

        rendered_depth = render_value(weights, depths.unsqueeze(-1))
        after_depth_mask = depths > rendered_depth
        idx_of_depth = after_depth_mask.int().argmax(-1)
        idxer = torch.arange(0, weights.shape[0])
        dual_surface_points =\
            (weights[idxer, idx_of_depth] < torch.finfo(sigma.dtype).eps) &\
            (weights[idxer, (idx_of_depth - 1).clamp(0, weights.shape[1] - 1)] < torch.finfo(sigma.dtype).eps)
        dsp = dual_surface_points.squeeze(-1)
        
        first_surface_alpha = torch.where(~after_depth_mask[dsp], weights[dsp].cumsum(-1), 0.0).amax(-1, keepdim=True)
        weights[dsp] = weights[dsp] / torch.where(first_surface_alpha > re_weigh_alpha, first_surface_alpha, 1.0)
        weights[dsp] = torch.where(~after_depth_mask[dsp], weights[dsp], 0.0)

    return weights


def render_value(weights: Tensor, values: Tensor) -> Tensor:
    """Performs Volumetric Rendering with weights for values

    Args:
        weights (shape[N, M] | shape[N, M, 1]): Render weight per sample point
        values (shape[N, M, K]): Values for sampled points

    Returns:
        rendered (shape[N, K]): Rendered value per ray
    """
    if weights.ndim != 3:
        weights = weights.unsqueeze(-1)
    return torch.sum(weights * values, dim=-2)
