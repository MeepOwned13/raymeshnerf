import torch
from torch import Tensor
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
import numpy as np

from .rays import create_rays, sobel_filter


def create_nerf_data(images: Tensor, c2ws: Tensor, focal: Tensor,
                     weight_epsilon: float = 0.33) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Creates rays for NeRF training

    Args:
        images (shape[N, H, W, 3]): Images to extract colors and sizes from
        c2ws (shape[N, 4, 4]): Extrinisic camera matrices (Camera to World)
        focal (shape[]): Focal length
        weight_epsilon: Added epsilon for pixel weights

    Returns:
        origins (shape[N * H * W, 3]): Ray origins in World coordinates
        directions (shape[N * H * W, 3]): Cartesian ray directions in World
        colors (shape[N * H * W, 3]): RGB colors for rays
        pixel_weights (shape[N * H * W]): Sampling edge weights for rays
    """
    origins = []
    directions = []
    colors = []

    # Collecting to list then concat for ease
    for image, c2w in zip(images, c2ws):
        intrinsic = torch.tensor([
            [focal.item(), 0, image.shape[1] // 2],
            [0, focal.item(), image.shape[0] // 2],
            [0, 0, 1],
        ], dtype=torch.float32)

        o, d = create_rays(image.shape[0], image.shape[1], intrinsic, c2w)

        origins.append(o.flatten(0, 1))
        directions.append(d.flatten(0, 1))
        colors.append(image.flatten(0, 1))

    origins = torch.cat(origins, dim=0)
    directions = torch.cat(directions, dim=0)
    colors = torch.cat(colors, dim=0)

    pixel_weights = sobel_filter(images).flatten() + weight_epsilon
    return origins, directions, colors, pixel_weights


class ImportantPixelSampler(WeightedRandomSampler):
    """Sampler implementing Importan Pixels Sampling for NeRF"""

    def __init__(self, weights: Tensor, num_samples: int, replacement: bool = True, swap_strategy_iter: int = 100,
                 step_epsilon: float = 1e-4, grouping_factor: int = 2 ** 24 -1):
        """Init

        Args:
            weights (shape[N]): Pixel weights assigned by edge detection
            num_samples: Number of samples to draw per __iter__ (epoch)
            replacement: Choose /w replacement?
            swap_strategy_iter: Specifies at which iteration Squared Error sampling takes over fully
            step_epsilon: Increase of weights not being chosen
        """
        super(ImportantPixelSampler, self).__init__(weights=weights,
                                                    num_samples=num_samples, replacement=replacement, generator=None)
        weights = weights.to(torch.float32)
        self.pixel_weights: Tensor = weights / weights.max()
        """(shape[N]) Pixel weights assigned by edge detection"""

        self.weights: Tensor = self.pixel_weights
        """(shape[N]) Weights used for choosing the next samples"""

        self.swap_strategy_iter: int = swap_strategy_iter
        """Specifies at which iteration Squared Error sampling takes over fully"""

        self.step_epsilon: float = step_epsilon
        """Increase of weights not being chosen"""

        self.num_iters: int = 0
        """Counts started iterations"""

        self.squared_errors: Tensor = torch.ones(self.pixel_weights.shape, dtype=torch.float32)
        """(shape[N]) Stores squared errors for pixels"""

        # Grouping needed since multinomial can only take 2**24 inputs
        groups = list(range(0, weights.shape[0], grouping_factor)) + [weights.shape[0]]
        self.groups = torch.tensor([
            (low, high, self.weights[low:high].sum()) for low, high in zip(groups[:-1], groups[1:])
        ], dtype=torch.float32)
        """(shape[num_groups, 3]) Stores low, high, group weight per row, required for
        sampling weights over size 2**24-1"""

    def __iter__(self):
        self.num_iters += 1
        group_sample = torch.multinomial(self.groups[..., -1], self.num_samples, replacement=True)
        groups, group_counts = group_sample.unique(sorted=False, return_counts=True)

        randoms = []
        for g, gc in zip(groups, group_counts):
            low, high, _ = self.groups[g]
            randoms.append(torch.multinomial(
                self.weights[int(low):int(high)], gc, self.replacement, generator=self.generator
            ))
        
        yield from iter(torch.cat(randoms).flatten().tolist())

    def update_errors(self, idxs: Tensor, errors: Tensor):
        """Update squared errors and weights for given indicies

        Args:
            idxs (shape[K]): Specifies which indicies to edit weights on
            errors (shape[K]): Freshly calculated squared errors for the idxs
        """
        errors = errors.clone().cpu().detach()
        # Epsilon evaluates to 5e-3 as the prev value contributes 20%
        self.squared_errors[idxs] = self.squared_errors[idxs] * 0.2 + errors * 0.8 + 4e-3  # Discounted error update
        self.weights += self.step_epsilon  # Increase weights of entries not chosen, needed to re-evaluate areas
        pxw = torch.clamp(torch.tensor([1.0], dtype=torch.float32) - self.num_iters / self.swap_strategy_iter, 0.0, 1.0)
        old_weights = self.weights[idxs].clone()
        self.weights[idxs] = self.pixel_weights[idxs] * pxw + self.squared_errors[idxs] * (1 - pxw)

        # Group wise update
        for i in range(self.groups.shape[0]):
            low, high, group_weight = self.groups[i]
            mask = (idxs >= low) & (idxs < high)
            group_weight += self.weights[idxs][mask].sum() - old_weights[mask].sum()
            self.groups[i, -1] = group_weight


def find_val_angles(c2ws: torch.Tensor, horizontal_partitions: int = 4, vertical_partitions: int = 2):
    """Deterministically get validation angle indicies from extrinsic camera matrices

    Args:
        c2ws (shape[N, 4, 4]): Extrinisic camera matrices (Camera to World)
        horizontal_partitions: #angles to get along the horizontal plane
        vertical_partitions: #angles to get along the vertical plane

    Returns:
        idxs (shape[horizontal_partitions * vertical_partitions]): Indicies of chosen validation angles
    """
    # Using spherical coordinates so choosing the middle of the partitions is easier
    positions = c2ws[:, :3, -1].clone()
    positions = F.normalize(positions, "fro", -1)
    cam_theta = torch.atan2(positions[..., 1], positions[..., 0])
    cam_phi = torch.arcsin(positions[..., 2])

    # Taking middle of partitions to find closest camera angle
    inclination_step = (cam_theta.max() - cam_theta.min()) / horizontal_partitions
    azimuth_step = (cam_phi.max() - cam_phi.min()) / vertical_partitions
    part_theta, part_phi = torch.meshgrid(
        torch.arange(cam_theta.min() + inclination_step / 2, cam_theta.max(), inclination_step),
        torch.arange(cam_phi.min() + azimuth_step / 2, cam_phi.max(), azimuth_step),
        indexing="ij"
    )

    part_theta, part_phi = part_theta.flatten().unsqueeze(0), part_phi.flatten().unsqueeze(0)
    cam_theta, cam_phi = cam_theta.unsqueeze(1), cam_phi.unsqueeze(1)
    # N,1 | 1,K -> N,K
    distances = torch.sqrt(
        2 - 2 * torch.sin(cam_theta) * torch.sin(part_theta) * torch.cos(cam_phi - part_phi) +
        torch.cos(cam_theta) * torch.cos(part_theta)
    )

    return torch.argmin(distances, dim=0)


def compute_near_far_planes(c2ws: Tensor) -> tuple[float, float]:
    """Compute minimal near and maximal far plane

    Transforms the box bounded by -1 to 1 in World coordinates to camera
    and finds the minimal near plane and maximal far plane based on distance

    Args:
        c2ws (shape[K, 4, 4]): Extrinsic camera matrices (Camera to World)

    Returns:
        near_plane: Minimal near plane found
        far_plane: Maximal far plane found
    """
    scene_bounds_min = torch.tensor([-1, -1, -1], dtype=torch.float32)
    scene_bounds_max = torch.tensor([1, 1, 1], dtype=torch.float32)

    # Transform bounding box corners to camera coordinates
    corners = torch.tensor([
        [scene_bounds_min[0], scene_bounds_min[1], scene_bounds_min[2]],
        [scene_bounds_min[0], scene_bounds_min[1], scene_bounds_max[2]],
        [scene_bounds_min[0], scene_bounds_max[1], scene_bounds_min[2]],
        [scene_bounds_min[0], scene_bounds_max[1], scene_bounds_max[2]],
        [scene_bounds_max[0], scene_bounds_min[1], scene_bounds_min[2]],
        [scene_bounds_max[0], scene_bounds_min[1], scene_bounds_max[2]],
        [scene_bounds_max[0], scene_bounds_max[1], scene_bounds_min[2]],
        [scene_bounds_max[0], scene_bounds_max[1], scene_bounds_max[2]],
    ])

    nears, fars = [], []
    for c2w in c2ws:
        corners_camera = (c2w[:3, :3] @ corners.T).T + c2w[:3, -1]
        distances = torch.norm(corners_camera, "fro", dim=1)
        nears.append(torch.min(distances))
        fars.append(torch.max(distances))

    near_plane = min(distances) * 0.9  # Slightly smaller than the closest point
    far_plane = max(distances) * 1.1  # Slightly larger than the farthest point

    return near_plane.item(), far_plane.item()


def load_npz(path: str) -> tuple[Tensor, Tensor, Tensor]:
    """Load numpy data

    Args:
        path: .npz file path

    Returns:
        images (shape[N, H, W, 3]): Images
        c2ws (shape[N, 4, 4]): Extrinisic camera matrices (Camera to World)
        focal (shape[]): Focal length
    """
    data = np.load(path)

    images = torch.from_numpy(data["images"]).to(torch.float32)
    c2ws = torch.from_numpy(data["poses"]).to(torch.float32)
    focal = torch.from_numpy(data["focal"]).to(torch.float32)

    return images, c2ws, focal

