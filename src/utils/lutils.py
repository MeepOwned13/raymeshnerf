import torch
from torch import Tensor
from torch.nn import MSELoss
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure,\
    learned_perceptual_image_patch_similarity
import lightning as L
from lightning.pytorch.callbacks import Callback
import warnings
import re
from random import random

from . import data, rays, colmap


class NeRFData(L.LightningDataModule):
    def __init__(self, name: str, source: data.ObjectSource, batch_size: int = 1024, val_angle_count: int | None = None,
                 val_angle_equidistant: bool = False, keep_val_in_train: bool = False):
        """Init

        Args:
            name: Name of object in data directory
            batch_size: #Rays in a batch
            val_angle_count: How many equidistant validation angles to choose
            val_angle_equidistant: Use equidistant val angles, or farthest point sampled ones (better for non 360)
            keep_val_in_train: Don't remove validation images from training set (useful for monitoring train images)
        """
        super().__init__()
        self.save_hyperparameters()
        # Making sure it is in the ObjectSource Enum for function calls
        self.hparams.source = data.ObjectSource(self.hparams.source)

    @property
    def scene_name(self):
        return f"{self.hparams.source.value}_{re.sub(r"\s+", r"_", self.hparams.name)}"

    def load_from_file(self):
        return data.load_data(self.hparams.name, self.hparams.source)

    def setup(self, stage: str):
        self.images, self.c2ws, self.intrinsics = self.load_from_file()

        # Swapping between automatic choice of "equidistant angles" and pre-set indices
        if self.hparams.val_angle_equidistant:
            val_idxs = data.find_val_angles_eq(c2ws=self.c2ws, angle_count=self.hparams.val_angle_count)
        else:
            val_idxs = data.find_rmn_angles(c2ws=self.c2ws, angle_count=self.hparams.val_angle_count)
        val_imgs = self.images[val_idxs]
        val_c2ws = self.c2ws[val_idxs]
        val_intrinsics = self.intrinsics[val_idxs]

        train_idxs = [i for i in range(self.images.shape[0]) if (i not in val_idxs) or self.hparams.keep_val_in_train]
        train_imgs = self.images[train_idxs]
        train_c2ws = self.c2ws[train_idxs]
        train_intrinsics = self.intrinsics[train_idxs]

        if stage == "fit":
            origins, directions, colors = data.create_nerf_data(train_imgs, train_c2ws, train_intrinsics)

            depths, errors = colmap.get_sparse_sfm_depths(train_imgs, train_c2ws, train_intrinsics)
            depths, errors = depths.flatten(0, -2), errors.flatten(0, -2)
            self.ds_mask = (errors != torch.inf).squeeze(-1)
            
            self.train_rays: TensorDataset = TensorDataset(
                origins, directions, colors, depths, errors
            )
            """Dataset: (origins, directions, colors, depths, errors)"""

            self.val_angles: TensorDataset = TensorDataset(
                val_c2ws,
                val_intrinsics,
                val_imgs
            )
            """Dataset: (c2w, intrinsic, image)"""

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_rays,
            batch_sampler=data.DSNeRFBatchSampler(self.ds_mask, batch_size=self.hparams.batch_size),
            num_workers=6,
            prefetch_factor=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_angles,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )


class LVolume(L.LightningModule):
    def __init__(self, **kwargs):
        """Init"""
        super().__init__()
        self.save_hyperparameters()
        self.nerf: torch.nn.Module = None

        self.background_noise_range = [0.4, 0.6]
        self.near_offset = -1.7320507764816284
        """Near offset, defaulted to scenes being contained in [-1, 1] bbox => radius sqrt(3) sphere"""
        self.far_offset = 1.7320507764816284
        """Far offset, defaulted to scenes being contained in [-1, 1] bbox => radius sqrt(3) sphere"""

    def setup(self, stage):
        if self.nerf is None:
            raise NotImplementedError(f"{self.__class__} must have .nerf attribute defined")
        if stage == "fit":
            self.lossf = MSELoss()
    
    def render_rays(self, origins: Tensor, directions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Render rays ready for display (e.g. don't return separate coarse, fine colors)

        Args:
            origins (shape[N, 3]): Ray origins in World coordinates
            directions (shape[N, 3]): Cartesian ray directions in World

        Returns:
            tuple: a tuple containing (rgb, depth, acc) where
            - **rgb**: *shape[N, 3]*: RGB value calculated for ray,
            - **depth**: *shape[N]*: Approximated depth of ray termination,
            - **acc**: *shape[N, 1]*: Sum of weights for pixel (alpha)
        """
        raise NotImplementedError(f"{self.__class__} hasn't implemented render_rays yet")
    
    def calculate_loss(self, origins: Tensor, directions: Tensor, colors: Tensor, sfm_depths: Tensor, sfm_errors: Tensor
                       ) -> Tensor:
        """Calculate loss for rays

        Args:
            origins (shape[N, 3]): Ray origins in World coordinates
            directions (shape[N, 3]): Cartesian ray directions in World
            colors (shape[N, 3]): Target pixel colors
            sfm_depths (shape[N]): SFM depth estimates (-1 where no estimates)
            sfm_errors (shape[N]): SFM depth errors (torch.inf where no estimates)

        Return:
            loss (shape[]): Loss
        """
        raise NotImplementedError(f"{self.__class__} hasn't implemented render_rays yet")

    @torch.no_grad()
    def render_image(self, height: int, width: int, c2w: Tensor, intrinsic: Tensor,
                     batch_size: int | None = None) -> Tensor:
        """Renders an image using NeRF and Volume Rendering

        Args:
            height: Image height
            width: Image width
            c2w (shape[4, 4]): Extrinsic camera matrix (Camera to World)
            intrinsic (shape[3, 3]): Intrinsic camera matrix
            batch_size: Batch size for rendering, if None uses hparams

        Returns:
            image (shape[height, width, 3]): Rendered image
        """
        batch_size = self.hparams.get("batch_size", batch_size) or self.trainer.datamodule.hparams.batch_size

        origins, directions = rays.create_rays(
            height=height,
            width=width,
            intrinsic=intrinsic,
            c2w=c2w
        )
        origins, directions = origins.flatten(0, -2), directions.flatten(0, -2)
        data = DataLoader(TensorDataset(origins, directions), batch_size=batch_size, shuffle=False)

        image = []
        for o, d in data:
            rgb, _, alpha = self.render_rays(
                origins=o,
                directions=d,
            )
            image.append(torch.cat((rgb, alpha), dim=-1))

        return torch.cat(image, 0).reshape(height, width, -1).clamp(0.0, 1.0)
    
    def locate_density_gradient_based_surface_depth(
        self, origins: Tensor, directions: Tensor, sigma_limit: float = 5.0, gamma: float = 5e-6,
        non_grad_step_size: float = 3e-2, min_step_size: float = 1e-5, max_iters: int = 300,
        silence_input_size_warning: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Main algorithm of RayMeshNerf, finds surface point depth using gradient ascent
        
        Args:
            coordinates (shape[..., in_coordinates]): Input point coordinates, in_coordinates specified by `volume`
            directions (shape[..., in_directions]): Input directions, in_directions specified by `volume`
            sigma_limit: When sigma is larger than this, we use gradient ascent
            gamma: Step size multiplier (analogous to learning rate for gradient descent)
            non_grad_step_size: Step size when sigma is below limit
            min_step_size: If step size falls below this, point is considered a surface point
            max_iters: Limit for iteration count
            silence_input_size_warning: Silence the warning associated with input count being over 2^19 for GPU

        Returns:
            tuple: tuple containing (depths, finish_mask)
            - **depths**: *shape[..., 1]*: Depths per origin-direction pair at the end of the search
            - **finish_mask**: *shape[...]*: Boolean mask indicating which positions depth was found before termination.
                True if: (1) gradient fell below grad_epsilon, i.e. surface point depth is located or
                (2) depth exceeds Model's far plane
        """
        shape = origins.shape[:-1]

        total_count = torch.prod(torch.tensor(shape)).item()
        if total_count > 2**19 and self.device != torch.device('cpu') and not silence_input_size_warning:
            warnings.warn(
                f"Total input count over {2**19} ({total_count}). This may cause inconsistencies with CUDA and ROCM "
                "implementations, resulting in the model returning the same numbers for points beyond the limit "
                "and thus calculating incorrect depths. Make sure to batch input to at most 2^19 chunks",
                category=UserWarning,
                stacklevel=2  # Shows the caller's line in the warning
            )

        origins, directions = origins.to(self.device), directions.to(self.device)

        dist_to_zero = torch.sqrt(torch.sum(torch.pow(origins, 2), -1, keepdim=True))
        depth, far_plane = (dist_to_zero + self.near_offset), (dist_to_zero + self.far_offset)
        
        in_progress_mask = torch.ones(shape, dtype=torch.bool, device=self.device)
        sigma_limit = torch.log(torch.tensor(sigma_limit)).item()

        with torch.no_grad():
            iters = 0
            while iters < max_iters:
                # Masking to save computation time (really effective at steps above (far-near)/non_grad_step_size
                #   as empty space gets explored in that many steps)
                masked_origins, masked_directions = origins[in_progress_mask], directions[in_progress_mask]

                # Depth needs masked cloning and detach first to avoid reference to full depth array
                #   (would result in inplace ops which break backward)
                masked_depth = depth[in_progress_mask].clone().detach()

                # delta_sigma/delta_depth calculated
                sigma = torch.log(
                    self.nerf(masked_origins + masked_depth * masked_directions, masked_directions, skip_colors=True)
                )
                sigma_nx = torch.log(
                    self.nerf(masked_origins + (masked_depth + 1e-6) * masked_directions, masked_directions, skip_colors=True)
                )
                grad = (sigma_nx - sigma) / 1e-6

                # fixed step size if sigma is below a limit (aka. empty space areas)
                grad_step = sigma > sigma_limit
                # Limiting step by offset to inch towards maxima if it would just jump back and forth
                step_size = (grad * non_grad_step_size * gamma).clamp(
                    -min_step_size * 100 + torch.rand(1, dtype=torch.float32, device=self.device) * min_step_size * 10,
                    min_step_size * 100 - torch.rand(1, dtype=torch.float32, device=self.device) * min_step_size * 10
                )
                step_size[~grad_step] = non_grad_step_size

                # Stepping stops if:
                # - gradient falls below limit (and this step used the gradient!), surface point depth estimate is found
                # - depth goes beyond the far plane
                mask_update = ((step_size.abs() >= min_step_size) | ~grad_step).squeeze(-1) &\
                            (masked_depth < far_plane[in_progress_mask]).squeeze(-1)
                # - already stopped at a previous step (ensured by re-indexing mask)
                in_progress_mask[in_progress_mask.clone()] = mask_update
                if (~in_progress_mask).all():
                    break
            
                # Updated to large depth array, update mask used to filter ones stopping in the current step
                depth[in_progress_mask] = masked_depth[mask_update] + step_size[mask_update]

                iters += 1
        
        return depth.detach(), ~in_progress_mask
    
    def estimate_normals(self, points: torch.Tensor, silence_input_size_warning: bool = False) -> torch.Tensor:
        """Estimate normals from points as the direction of largest negative gradient
        
        Args:
            points (shape[N, 3]): XYZ coordinates of points
            silence_input_size_warning: Silence the warning associated with input count being over 2^19 for GPU

        Returns:
            normals (shape[N, 3]): Estimated normal vectors
        """

        total_count = torch.prod(torch.tensor(points.shape[:-1])).item()
        if total_count > 2**19 and self.device != torch.device('cpu') and not silence_input_size_warning:
            warnings.warn(
                f"Total input count over {2**19} ({total_count}). This may cause inconsistencies with CUDA and ROCM "
                "implementations, resulting in the model returning the same numbers for points beyond the limit "
                "and thus calculating incorrect normals. Make sure to batch input to at most 2^19 chunks",
                category=UserWarning,
                stacklevel=2  # Shows the caller's line in the warning
            )

        coords: torch.Tensor = points.clone().to(self.device)
        coords.requires_grad = True
        coords.retain_grad()

        sigmas = self.nerf(coords, None, skip_colors=True)
        sigmas.backward(torch.ones_like(sigmas))

        normals = -torch.nn.functional.normalize(coords.grad, p="fro", dim=-1)
        return normals

    def training_step(self, batch, batch_idx):
        origins, directions, colors, depths, errors = batch
        loss = self.calculate_loss(origins, directions, colors, depths, errors)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_imgs = []

    def validation_step(self, batch, batch_idx):
        c2w, intrinsic, image = batch
        render = self.render_image(image.shape[1], image.shape[2], c2w[0], intrinsic[0]).unsqueeze(0)
        
        cloned_render = render.clone()  # Used for display
        self.val_imgs.append(cloned_render.permute(0, 3, 1, 2))

        # Transparency isn't handled well by PSNR, compositing with neutral gray background
        if image.shape[-1] == 4:  # Image side
            background = torch.full_like(image[..., :3], 0.5)
            image = (image[..., :3] * image[..., 3:4]) + (background * (1 - image[..., 3:4]))
        
        if render.shape[-1] == 4:  # Render side
            background = torch.full_like(render[..., :3], 0.5)
            render = (render[..., :3] * render[..., 3:4]) + (background * (1 - render[..., 3:4]))

        render, image = render.permute(0, 3, 1, 2), image.permute(0, 3, 1, 2)
        psnr = peak_signal_noise_ratio(render, image, data_range=(0.0, 1.0))
        ssim = structural_similarity_index_measure(render, image, data_range=(0.0, 1.0))
        lpips = learned_perceptual_image_patch_similarity(render, image, normalize=True)
        metrics = {"val_psnr": psnr, "val_ssim": ssim, "val_lpips": lpips}
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)
        return metrics

    def on_validation_epoch_end(self):
        if self.trainer and not self.trainer.sanity_checking:  # Disable image logging on sanity check
            images = torch.cat(self.val_imgs, dim=0)
            self.logger.experiment.add_image("Renders", make_grid(images, nrow=4, padding=5), self.global_step)
        self.val_imgs.clear()

    def configure_optimizers(self):
        raise NotImplementedError("configure_optimizers must be overwritten in subclass")
    

class OGFilterCallback(Callback):
    def __init__(self, per_backwards: int = 8, full_update_n_backwards: int = 8):
        """Callback to update Occupancy Grid filter of trainer.nerf
        
        Args:
            per_backwards: Update filter after n backward operations
            full_update_n_backwards: First n backwards to use full update for, selection of grid nodes is stochastic
                afterwards
        """
        self.per_backwards = per_backwards
        self.full_update_n_backwards = full_update_n_backwards
    
    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        gstep = trainer.global_step
        if self.per_backwards and gstep > 0 and (gstep % self.per_backwards == 0):
            pl_module.nerf.update_filter(full_selection=gstep <= (self.full_update_n_backwards * self.per_backwards))
        return super().on_before_zero_grad(trainer, pl_module, optimizer)
