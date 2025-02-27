import torch
from torch import Tensor
from torch.utils.data import WeightedRandomSampler
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

    def __init__(self, weights: Tensor, num_samples: int, replacement: bool = True, swap_strategy_iter: int = 100):
        """Init

        Args:
            weights (shape[N]): Pixel weights assigned by edge detection
            num_samples: Number of samples to draw per __iter__ (epoch)
            replacement: Choose /w replacement?
            swap_strategy_iter: Specifies at which iteration Squared Error sampling takes over fully
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

        self.num_iters: int = 0
        """Counts started iterations"""

        self.squared_errors: Tensor = torch.ones(self.pixel_weights.shape, dtype=torch.float32)
        """(shape[N]) Stores squared errors for pixels"""

    def __iter__(self):
        self.num_iters += 1
        yield from super(ImportantPixelSampler, self).__iter__()

    def update_errors(self, idxs: Tensor, errors: Tensor):
        """Update squared errors and weights for given indicies

        Args:
            idxs (shape[K]): Specifies which indicies to edit weights on
            errors (shape[K]): Freshly calculated squared errors for the idxs
        """
        errors = errors.clone().cpu().detach()
        # Epsilon evaluates to 5e-3 as the prev value contributes 20%
        self.squared_errors[idxs] = self.squared_errors[idxs] * 0.2 + errors * 0.8 + 4e-3  # Discounted error update
        pxw = torch.clamp(torch.tensor([1.0], dtype=torch.float32) - self.num_iters / self.swap_strategy_iter, 0.0, 1.0)
        self.weights[idxs] = self.pixel_weights[idxs] * pxw + self.squared_errors[idxs] * (1 - pxw)


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

