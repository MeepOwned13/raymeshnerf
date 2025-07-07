import torch
from torch import Tensor
from torch.nn import functional as F
from torch import Tensor

from .lutils import LVolume


# In accordance with mitsuba's conventions
def look_at(radius: float, theta: Tensor, phi: Tensor) -> Tensor:
    """Construct Look At matrix for [0, 0, 0] target

    Args:
        radius: Distance to target, aka. radius of sphere the camera is laying on
        theta shape([]): Vertical rotation in radians
        phi shape([]): Horizontal rotation in radians

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
    phis = torch.pi * i * (1 + torch.sqrt(torch.tensor(5)))
    thetas = torch.arccos(1 - 2 * i / n)

    return phis, thetas


def locate_density_gradient_based_surface_depth(
    volume: LVolume, origins: Tensor, directions: Tensor, sigma_limit: float = 5.0, gamma: float = 5e-6,
    non_grad_step_size: float = 3e-2, grad_epsilon: float = 5e-2, max_iters: int = 300
) -> tuple[Tensor, Tensor]:
    """Main algorithm of RayMeshNerf, finds surface point depth using gradient ascent
    
    Args:
        volume: Model used to evaluate density for 3D points
        coordinates (shape[..., in_coordinates]): Input point coordinates, in_coordinates specified by `volume`
        directions (shape[..., in_directions]): Input directions, in_directions specified by `volume`
        sigma_limit: When sigma is larger than this, we use gradient ascent
        gamma: Step size multiplier (analogous to learning rate for gradient descent)
        non_grad_step_size: Step size when sigma is below limit
        grad_epsilon: If the magnitude of the gradient falls below this, we found depth for the surface point
        max_iters: Limit for iteration count

    Returns:
        tuple: tuple containing (depths, finish_mask)
        - **depths**: *shape[..., 1]*: Depths per origin-direction pair at the end of the search
        - **finish_mask**: *shape[...]*: Boolean mask indicating which positions depth was found before termination.
            True if: (1) gradient fell below grad_epsilon, i.e. surface point depth is located or
            (2) depth exceeds Model's far plane
    """
    shape = origins.shape[:-1]
    depth = torch.full(shape + (1,), volume.hparams.near, dtype=torch.float32, device=volume.device)
    in_progress_mask = torch.ones(shape, dtype=torch.bool, device=volume.device)
    origins, directions = origins.to(volume.device), directions.to(volume.device)

    iters = 0
    while iters < max_iters:
        # Masking to save computation time (really effective at steps above (farplane-nearplane)/non_grad_step_size)
        masked_origins, masked_directions = origins[in_progress_mask], directions[in_progress_mask]

        # Depth needs masked cloning and detach first to avoid reference to full depth array
        #   (would result in inplace ops which break backward)
        masked_depth = depth[in_progress_mask].clone().detach()
        masked_depth.requires_grad = True  # Re-enabling grad to find depth gradient
        masked_depth.retain_grad()  # Needed to retain grad for non-leaf nodes in the computation graph

        # delta_sigma/delta_depth calculated
        sigma = volume.nerf(masked_origins + masked_depth * masked_directions, masked_directions, skip_colors=True)
        sigma.backward(torch.ones_like(sigma))  # Backward without loss function

        # fixed step size if sigma is below a limit (aka. empty space areas)
        grad_step = sigma > sigma_limit
        grad = masked_depth.grad
        step_size = grad * gamma
        step_size[~grad_step] = non_grad_step_size

        # Stepping stops if:
        # - gradient falls below limit (and this step used the gradient!), as the surface point depth estimate is found
        # - depth goes beyond the far plane
        mask_update = ((grad.abs() >= grad_epsilon) | ~grad_step).squeeze(-1) &\
                      (masked_depth < volume.hparams.far).squeeze(-1)
        # - already stopped at a previous step (ensured by re-indexing mask)
        in_progress_mask[in_progress_mask.clone()] = mask_update
        if (~in_progress_mask).all():
            break

        # Updated to large depth array, update mask used to filter ones stopping in the current step
        depth[in_progress_mask] = masked_depth[mask_update] + step_size[mask_update]

        iters += 1

    return depth, ~in_progress_mask
