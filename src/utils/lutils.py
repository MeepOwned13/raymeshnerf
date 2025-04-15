import torch
from torch import Tensor
from torch.nn import MSELoss
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.image import PeakSignalNoiseRatio
import lightning as L
from lightning.pytorch.callbacks import Callback
import random
import numpy as np

from . import data, rays


def w_init_fn(worker_id):
    # Get current random seed
    worker_seed = torch.initial_seed() % 2**32
    # Set seeds for all relevant libraries
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    # For CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)


class NeRFData(L.LightningDataModule):
    def __init__(self, object_name: str, batch_size: int = 1024, horizontal_val_angles: int = 4,
                 vertical_val_angles: int = 3, epoch_size: int = 2**20, rays_per_image: int = 2**12,
                 subpixel_sampling: bool = False):
        """Init

        Args:
            path: Path to data file
            batch_size: Batch size and epoch size
            horizontal_val_angles: #angles to take for validation horizontally
            vertical_val_angles: #angles to take for validation vertically
            swap_strategy_iter: Specifies at which iteration Squared Error sampling takes over fully per image
        """
        super().__init__()
        self.save_hyperparameters()
        data.RayDataset.disable_multiprocessing_length_warning()

    def load_from_file(self):
        return data.load_obj_data(self.hparams.object_name)

    def setup(self, stage: str):
        images, c2ws, focal = self.load_from_file()
        self.hparams.near, self.hparams.far = data.compute_near_far_planes(c2ws=c2ws)
        self.hparams.focal = focal.item()
        self.save_hyperparameters()

        val_idxs = data.find_val_angles(
            c2ws=c2ws,
            horizontal_partitions=self.hparams.horizontal_val_angles,
            vertical_partitions=self.hparams.vertical_val_angles,
        )
        val_imgs, val_c2ws = images[val_idxs], c2ws[val_idxs]

        train_idxs = [i for i in range(images.shape[0]) if i not in val_idxs]
        train_imgs, train_c2ws = images[train_idxs], c2ws[train_idxs]

        if stage == "fit":
            self.train_rays: data.RayDataset = data.RayDataset(
                images=train_imgs,
                c2ws=train_c2ws,
                focal=torch.tensor(self.hparams.focal, dtype=torch.float32),
                rays_per_image=self.hparams.rays_per_image,
                length=self.hparams.epoch_size,
                subpixel_sampling=self.hparams.subpixel_sampling,
            )
            """Dataset: (pointers, origins, directions, colors)"""

            self.val_angles: TensorDataset = TensorDataset(
                val_c2ws,
                torch.tensor([focal], dtype=torch.float32).expand(val_c2ws.shape[0]),
                val_imgs
            )
            """Dataset: (c2w, focal, image)"""

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_rays,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            worker_init_fn=w_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_angles,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            worker_init_fn=w_init_fn
        )


class LVolume(L.LightningModule):
    def __init__(self, coarse_samples: int = 64, fine_samples: int = 128, **kwargs):
        """Init

        Args:
            coarse_samples: Initial samples to take per ray
            fine_samples: Hierarchical resampling sample count
        """
        super().__init__()
        self.save_hyperparameters()
        self.nerf: torch.nn.Module = None

        self.lossf = MSELoss(reduction='none')
        self.psnr = PeakSignalNoiseRatio()

    def setup(self, stage):
        if self.nerf is None:
            raise NotImplementedError(f"{self.__class__} must have .nerf attribute defined")
        return super().setup(stage)

    def compute_along_rays(self, origins: Tensor, directions: Tensor, near: float | None = None,
                           far: float | None = None, coarse_samples: int | None = None, fine_samples: int | None = None,
                           deterministic: bool = True, **kwargs) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Uniformally and Hierarchically sample rays and calculate RGBS using NeRF

        Args:
            origins (shape[N, 3]): Ray origins in World coordinates
            directions (shape[N, 3]): Cartesian ray directions in World
            near: Near plane, the first sample points' depth, if None uses hparams
            far: Far plane, the last sample points' depth, if None uses hparams
            coarse_samples: Uniform sample count along rays, if None uses hparams
            fine_samples: Hierarchical sample count along rays, if None uses hparams
            deterministic: Should hierarchical sampling be deterministic?

        Returns:
            coarse_rgbs (shape[N, coarse_samples, 4]): RGBS predicted by NeRF for uniform samples
            coarse_depths (shape[N, coarse_samples]): Depths sampled uniformally, sorted and aligned to coarse_rgbs
            fine_rgbs (shape[N, fine_samples, 4]): RGBS predicted by NeRF for hierarchical samples
            fine_depths (shape[N, fine_samples]): Depths sampled hierarchically, sorted and aligned to fine_rgbs
        """
        near = self.hparams.get("near", near) or self.trainer.datamodule.hparams.near
        far = self.hparams.get("far", far) or self.trainer.datamodule.hparams.far
        coarse_samples = coarse_samples or self.hparams.coarse_samples
        fine_samples = fine_samples or self.hparams.fine_samples

        # This function deviates from the original NeRF paper as coarse and fine samples are processed by the same model
        points, expanded_directions, coarse_depths = rays.sample_ray_uniformally(
            origins=origins,
            directions=directions,
            near=near,
            far=far,
            num_samples=coarse_samples,
        )
        coarse_rgbs = self.nerf(points, expanded_directions)

        # Bin bounds are halfway between sampled coordinates + near + far plane
        bins = torch.cat([
            torch.tensor(near, dtype=torch.float32, device=self.device).expand(origins.shape[0], 1),
            (coarse_depths[..., 1:] + coarse_depths[..., :-1]) / 2,
            torch.tensor(far, dtype=torch.float32, device=self.device).expand(origins.shape[0], 1),
        ], -1)

        points, expanded_directions, fine_depths = rays.sample_ray_hierarchically(
            origins=origins,
            directions=directions,
            num_samples=fine_samples,
            bins=bins,
            weights=coarse_rgbs[..., -1],
            deterministic=deterministic,
        )
        fine_rgbs = self.nerf(points, expanded_directions)

        # deterministic ensures depth sorted output, if non-deterministic,
        # sort manually as sortedness is required for volume rendering
        if not deterministic:
            fine_depths, idxs = torch.sort(fine_depths, dim=-1)
            fine_rgbs = fine_rgbs[torch.arange(idxs.shape[0]).unsqueeze(1), idxs]

        return coarse_rgbs, coarse_depths, fine_rgbs, fine_depths

    @torch.no_grad()
    def render_image(self, height: int, width: int, c2w: Tensor, focal: Tensor, near: float | None = None,
                     far: float | None = None, batch_size: int | None = None) -> Tensor:
        """Renders an image using NeRF and Volume Rendering

        Args:
            height: Image height
            width: Image width
            c2w (shape[4, 4]): Extrinsic camera matrix (Camera to World)
            focal (shape[]): Focal length
            near: Near plane, the first sample points' depth, if None uses hparams
            far: Far plane, the last sample points' depth, if None uses hparams
            batch_size: Batch size for rendering, if None uses hparams

        Returns:
            image (shape[height, width, 3]): Rendered image
        """
        near = self.hparams.get("near", near) or self.trainer.datamodule.hparams.near
        far = self.hparams.get("far", far) or self.trainer.datamodule.hparams.far
        batch_size = self.hparams.get("batch_size", batch_size) or self.trainer.datamodule.hparams.batch_size

        intrinsic = torch.tensor([
            [focal.item(), 0, width // 2],
            [0, focal.item(), height // 2],
            [0, 0, 1],
        ], dtype=torch.float32, device=self.device)
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
            _, _, rgbs, depths = self.compute_along_rays(o, d, near, far)
            rgb, _, alpha = rays.render_rays(rgbs, depths)
            image.append(torch.cat((rgb, alpha), dim=-1))

        return torch.cat(image, 0).reshape(height, width, -1)

    def training_step(self, batch, batch_idx):
        pointers, origins, directions, colors = batch
        coarse_rgbs, coarse_depths, fine_rgbs, fine_depths = self.compute_along_rays(origins, directions)

        coarse_colors, _, coarse_alphas = rays.render_rays(rgbs=coarse_rgbs, depths=coarse_depths)
        fine_colors, _, fine_alphas = rays.render_rays(rgbs=fine_rgbs, depths=fine_depths)

        if colors.shape[-1] == 4:  # RGBA, apply background noise to ensure 0 density background
            colors, alphas = colors[..., :3], colors[..., 3:4]
            noise = torch.empty_like(colors).uniform_(0.0, 1.0)

            mixed_colors = colors * alphas + noise * (1 - alphas)
            mixed_coarse_colors = coarse_colors * coarse_alphas + noise * (1 - coarse_alphas)
            mixed_fine_colors = fine_colors * fine_alphas + noise * (1 - fine_alphas)

            loss = (
                self.lossf(mixed_coarse_colors, mixed_colors) + self.lossf(mixed_fine_colors, mixed_colors)
            ).mean(-1)
        else:  # RGB
            loss = (self.lossf(coarse_colors, colors) + self.lossf(fine_colors, colors)).mean(-1)

        self.trainer.datamodule.train_rays.update_weights(pointers, loss)
        loss = loss.mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_imgs = []

    def validation_step(self, batch, batch_idx):
        c2w, focal, image = batch
        render = self.render_image(image.shape[1], image.shape[2], c2w[0], focal[0]).unsqueeze(0)
        if image.shape[-1] == 4:  # Transparency isn't handled well by PSNR
            image = image[..., :3] * image[..., 3:4]
            render = render[..., :3] * render[..., 3:4]

        render, image = render.permute(0, 3, 1, 2), image.permute(0, 3, 1, 2)
        psnr = self.psnr(render, image)

        metrics = {"val_psnr": psnr}
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        self.val_imgs.append(render)
        return metrics

    def on_validation_epoch_end(self):
        if self.trainer and not self.trainer.sanity_checking:  # Disable image logging on sanity check
            images = torch.cat(self.val_imgs, dim=0)
            self.logger.experiment.add_image("Renders", make_grid(images, nrow=4, padding=5), self.global_step)
        self.val_imgs.clear()

    def configure_optimizers(self):
        raise NotImplementedError("configure_optimizers must be overwritten in subclass")


class OGFilterCallback(Callback):
    def __init__(self, num_backprops: int = 8):
        self.num_backprops = num_backprops
        self._current = 0

    def on_after_backward(self, _, module):
        self._current += 1
        if self._current >= self.num_backprops:
            module.nerf.update_filter()
            self._current = 0


class PixelSamplerUpdateCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer and not trainer.sanity_checking:  # Disable image logging on sanity check
            trainer.datamodule.train_rays.update_image_weights()

            weights = [trainer.datamodule.train_rays.data[i][3].weights for i in list(range(8))]
            weights = [w / w.max() for w in weights]
            weights = torch.stack(weights, dim=0).unsqueeze(1).expand(-1, 3, -1, -1)
            trainer.logger.experiment.add_image(
                "Sample weights", make_grid(weights, nrow=4, padding=5), trainer.global_step
            )
        return super().on_validation_epoch_end(trainer, pl_module)
