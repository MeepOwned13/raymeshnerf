import torch
from torch import Tensor, nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


def create_rays(height: int, width: int, intrinsic: Tensor, c2w: Tensor) -> tuple[Tensor, Tensor]:
    """Create rays cast by camera in World coordinates

    Args:
        height: Rays to take vertically (image height)
        width: Rays to take horizontally (image width)
        intrinsic (shape[3, 3]): Intrinsic camera matrix
        c2w (shape[4, 4]): Extrinsic camera matrix (Camera to World)

    Returns:
        ray_origins (shape[width, height, 3]): Ray origins in World coordinates
        ray_directions (shape[width, height, 3]): Cartesian ray directions in World
    """
    device = c2w.device

    focal_x = intrinsic[0, 0]
    focal_y = intrinsic[1, 1]
    # cx and cy handle the misalignement of the principal point with the center of the image
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # Index each point on the image, determine ray directions to them
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        indexing='xy'
    )
    directions = torch.stack((
        (i - cx) / focal_x,
        -(j - cy) / focal_y,
        -torch.ones(i.shape, dtype=torch.float32, device=device)  # -1 since ray is cast away from camera
    ), -1)

    # Transform ray directions to World, origins just need to be broadcasted accordingly
    ray_directions = F.normalize(directions @ c2w[:3, :3].T, "fro", -1)
    ray_origins = torch.broadcast_to(c2w[:3, -1], ray_directions.shape)  # c2w last column determines position

    return ray_origins, ray_directions


@torch.no_grad()
def sobel_filter(images: Tensor) -> Tensor:
    """Applies the Sobel-Feldman operator to a batch of images

    Args:
        images (shape[K, W, H, 3-4]): Batch of rgb(a) images

    Return:
        edges (shape[K, W, H]): Edge intensities
    """
    # Sobel-Feldman operator
    filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1,
                       padding=1, padding_mode='zeros', bias=False, dtype=torch.float32)
    gx = torch.tensor([
        [3.0, 0.0, -3.0],
        [10.0, 0.0, -10.0],
        [3.0, 0.0, -3.0],
    ], dtype=torch.float32)
    gy = torch.tensor([
        [3.0, 10.0, 3.0],
        [0.0, 0.0, 0.0],
        [-3.0, -10.0, -3.0],
    ], dtype=torch.float32)
    weights = torch.stack([gx, gy], 0).unsqueeze(1)
    filter.weight = nn.Parameter(weights, requires_grad=False)

    edges = filter(images[..., :3].mean(dim=-1).unsqueeze(1))
    edges = torch.sqrt(torch.sum(edges ** 2, dim=1))
    return edges


def sample_ray_uniformally(origins: Tensor, directions: Tensor, near: float, far: float,
                           num_samples: int, perturb=True) -> tuple[Tensor, Tensor, Tensor]:
    """Uniformally sample rays and return them in the World coordinate system

    Args:
        origins (shape[N, 3]): Ray origins in World coordinates
        directions (shape[N, 3]): Cartesian ray directions in World
        near: Near plane, the first sample points' depth
        far: Far plane, the last sample points' depth
        num_samples: How many samples to take along the ray
        perturb: If True, adds noise to the depths

    Returns:
        points (shpe[N, num_samples, 3]): Sampled points in World coordinates
        directions (shape[N, num_samples, 3]): Original directions expanded to match the shape of points
        depths (shape[N, num_samples]): Depth of each sampled point on the given ray
    """
    device = origins.device
    depths = torch.linspace(near, far, num_samples, dtype=torch.float32, device=device).expand(origins.shape[0], -1)

    if perturb:
        # Noise is at most half of step size, this ensures sorted depths, required for volume rendering
        noise = (torch.rand(depths.shape, device=device) - 0.5) * (far - near) / num_samples / 2
        # Clamping to stay between near and far
        depths = (depths + noise).clamp(near, far)

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
        points (shpe[N, num_samples, 3]): Sampled points in World coordinates
        directions (shape[N, num_samples, 3]): Original directions expanded to match the shape of points
        depths (shape[N, num_samples]): Depth of each sampled point on the given ray
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


def render_rays(rgbs: Tensor, depths: Tensor, white_background: bool = False) -> tuple[Tensor, Tensor]:
    """Performs Volumetric Rendering

    Args:
        rgbs (shape[N, M, 4]): RGB and Sigma values for sampled points
        depths (shape[N, M]): Specifies how far along the rays are the RGBSs
        white_background: If background is white, model output rgb is inverted

    Returns:
        rgb (shape[N, 3]): RGB value calculated for ray
        depth (shape[N]): Approximated depth of ray termination
        acc (shape[N, 1]): Sum of weights for pixel (alpha)
    """
    device = rgbs.device

    distances = depths[..., 1:] - depths[..., :-1]
    # 1e10 ensures the last color is rendered no matter what
    distances = torch.cat([distances, torch.tensor([1e10], device=device).expand(distances[..., :1].shape)], -1)
    # directions already normalized at ray calculation, so distances correspond to world already

    alpha = 1.0 - torch.exp(-F.relu(rgbs[..., 3]) * distances)
    # 1e10 ensures the last color is rendered no matter what
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1
    )[:, :-1]

    rgb = torch.sum(weights[..., None] * rgbs[..., :3], dim=-2)
    if white_background:
        rgb = 1 - rgb
    depth = torch.sum(weights * depths, dim=-1)
    acc = torch.sum(weights, dim=-1).unsqueeze(-1)

    return rgb, depth, acc

