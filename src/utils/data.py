import torch
from torch import Tensor
from torch.utils.data import WeightedRandomSampler, IterableDataset, RandomSampler
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import deque
from typing import Iterator

from .rays import create_rays, sobel_filter
from .mesh_render import render_mesh


def create_nerf_data(images: Tensor, c2ws: Tensor, focal: Tensor,
                     skip_cat: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Creates rays for NeRF training

    Args:
        images (shape[N, H, W, 3-4]): Images to extract colors and sizes from
        c2ws (shape[N, 4, 4]): Extrinisic camera matrices (Camera to World)
        focal (shape[]): Focal length
        weight_epsilon: Added epsilon for pixel weights
        skip_cat: Skips concatenation resulting in list of images (length N)

    Returns:
        origins (shape[N * H * W, 3]): Ray origins in World coordinates
        directions (shape[N * H * W, 3]): Cartesian ray directions in World
        colors (shape[N * H * W, 3-4]): RGB(A) colors for rays
        pixel_weights (shape[N * H * W]): Sampling edge weights for rays
    """
    origins = []
    directions = []
    colors = []
    pixel_weights = []

    # Collecting to list then concat for ease
    for image, c2w in zip(images, c2ws):
        intrinsic = torch.tensor([
            [focal.item(), 0, image.shape[1] // 2],
            [0, focal.item(), image.shape[0] // 2],
            [0, 0, 1],
        ], dtype=torch.float32)

        o, d = create_rays(image.shape[0], image.shape[1], intrinsic, c2w)
        weights = sobel_filter(image.unsqueeze(0)).squeeze(0) + 0.1

        origins.append(o.flatten(0, 1))
        directions.append(d.flatten(0, 1))
        colors.append(image.flatten(0, 1))
        pixel_weights.append(weights.flatten(0, 1))

    if not skip_cat:
        origins = torch.cat(origins, dim=0)
        directions = torch.cat(directions, dim=0)
        colors = torch.cat(colors, dim=0)
        pixel_weights = torch.cat(pixel_weights, dim=0)

    return origins, directions, colors, pixel_weights


class ImportantPixelSampler(WeightedRandomSampler):
    """Sampler implementing Important Pixels Sampling for NeRF"""

    def __init__(self, weights: Tensor, num_samples: int):
        """Init

        Args:
            weights (shape[N]): Pixel weights assigned by edge detection
            num_samples: Number of samples to draw per __iter__ (epoch)
            swap_strategy_iter: Specifies at which iteration Squared Error sampling takes over fully
        """
        super(ImportantPixelSampler, self).__init__(weights=weights, num_samples=num_samples,
                                                    replacement=False, generator=None)
        weights = weights.to(torch.float32)
        self.weights: Tensor = torch.clamp(weights / weights.max() + 1e-2, 0.0, 1.0)
        """(shape[N]) Weights used for choosing the next samples"""

        self.errors: Tensor = torch.ones_like(self.weights) * 0.1
        """(shape[N]) Stores squared errors for pixels"""

    def __iter__(self):
        yield from super(ImportantPixelSampler, self).__iter__()

    def update_errors(self, idxs: Tensor, errors: Tensor):
        """Updates errors for given indices

        Args:
            idxs (shape[K]): Specifies which indicies to edit weights on
            errors (shape[K]): Freshly calculated squared errors for the idxs
        """        
        idxs = idxs.cpu().detach()
        errors = errors.cpu().detach()
        self.errors[idxs] = errors


    def update_weights(self):
        """Applies errors to weights with normalization"""
        self.weights = self.errors.clone() / self.errors.max()
        # Reset errors
        self.errors: Tensor = torch.ones_like(self.weights) * self.weights.mean()


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
    c2ws = torch.from_numpy(data["c2ws"]).to(torch.float32)
    focal = torch.from_numpy(data["focal"]).to(torch.float32)

    return images, c2ws, focal


def load_obj_data(obj_name: str, sensor_count: int = 64, directory: str | None = None,
                  verbose: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    """Loads object data from disk, or renders if doesn't exist, follows Google Scanned Objects mesh format

    Args:
        obj_name: Name of object directory under directory
        sensor_count: Number of view angles to render if rendering is required
        directory: Directory to search objects under, defaults to project_root/data
        verbose: Print rendering info

    Returns:
        images (shape[N, H, W, 3]): Images
        c2ws (shape[N, 4, 4]): Extrinisic camera matrices (Camera to World)
        focal (shape[]): Focal length
    """
    directory = directory or f"{__file__}/../../../data"

    npz_path: Path = (Path(directory) / f"{obj_name}.npz").resolve().absolute()
    if not npz_path.exists():
        obj_path: Path = (Path(directory) / "raw_objects" / obj_name).resolve().absolute()
        if not obj_path.is_dir():
            raise ValueError(f"Directory of object '{obj_name}' doesn't exist")

        if verbose:
            print(f"Render of object '{obj_name}' not found, rendering {sensor_count} angles")

        images, c2ws, focal = render_mesh(
            obj_path=obj_path,
            sensor_count=sensor_count
        )
        np.savez_compressed(npz_path, images=images, c2ws=c2ws, focal=focal)

        if verbose:
            print(f"Render of '{obj_name}' complete")

    return load_npz(npz_path)


class RayDataset(IterableDataset):
    """Dataset that samples rays by image, images and rays inside them are weighted"""

    def __init__(self, images: Tensor, c2ws: Tensor, focal: Tensor, rays_per_image: int, length: int):
        """Init

        Args:
            images (shape[N, H, W, 3]): Images
            c2ws (shape[N, 4, 4]): Extrinisic camera matrices (Camera to World)
            focal (shape[]): Focal length
            rays_per_image: Samples per image
            length: Length Ligthning will use for epochs
        """
        super(RayDataset, self).__init__()
        self.rays_per_image = rays_per_image
        self.current_ray_idxs: deque = deque([])
        self.current_image_idx: int = None
        self._length = length

        origins, directions, colors, pixel_weights = create_nerf_data(images, c2ws, focal, skip_cat=True)

        self.data = [(
            o, d, c, ImportantPixelSampler(p, num_samples=self.rays_per_image)
        ) for o, d, c, p in zip(origins, directions, colors, pixel_weights)]

        self.image_weights = torch.tensor([d[3].weights.sum() for d in self.data])

    def __len__(self):
        return self._length

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        """Iterate dataset

        rays_per_image samples are taken from a single image and yielded 1 by 1, after which a new image is chosen

        Yields:
            origins (shape[3]): Ray origin in World coordinates
            directions (shape[3]): Cartesian ray direction in World
            colors (shape[3]): RGB colors for ray
        """
        while True:
            if not self.current_ray_idxs:
                self.current_image_idx = torch.multinomial(self.image_weights, num_samples=1).item()
                o, d, c, sampler = self.data[self.current_image_idx]
                self.current_ray_idxs = deque([i for i in sampler])

            idx = self.current_ray_idxs.pop()
            pointer = torch.tensor([self.current_image_idx, idx], dtype=int)
            yield pointer, o[idx], d[idx], c[idx]

    def update_errors(self, pointers: Tensor, errors: Tensor):
        """Update errors with pointers

        Args:
            pointers (shape[K, 2]): Specifies which samplers and indices to edit weights on
            errors (shape[K]): Freshly calculated squared errors for the pointers
        """
        for v in torch.unique(pointers[:, 0]):
            mask = pointers[:, 0] == v
            sampler: ImportantPixelSampler = self.data[v][3]
            sampler.update_errors(pointers[mask][:, 1], errors[mask])

    def update_weights(self):
        """Update sampler weights"""
        for i in range(len(self.data)):
            sampler = self.data[i][3]
            sampler.update_weights()
            self.image_weights[i] = sampler.weights.sum()
