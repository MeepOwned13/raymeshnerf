import torch
from torch import Tensor
from torch.nn import MSELoss
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.image import PeakSignalNoiseRatio
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
import warnings

import utils as U


class NeRFData(L.LightningDataModule):
    def __init__(self, object_name: str, batch_size: int = 1024, horizontal_val_angles: int = 4,
                 vertical_val_angles: int = 3, epoch_size: int = 2**20, rays_per_image: int = 2**12):
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

    def load_from_file(self):
        return U.data.load_obj_data(self.hparams.object_name)

    def setup(self, stage: str):
        images, c2ws, focal = self.load_from_file()
        self.hparams.near, self.hparams.far = U.data.compute_near_far_planes(c2ws=c2ws)
        self.hparams.focal = focal.item()
        self.save_hyperparameters()

        val_idxs = U.data.find_val_angles(
            c2ws=c2ws,
            horizontal_partitions=self.hparams.horizontal_val_angles,
            vertical_partitions=self.hparams.vertical_val_angles,
        )
        val_imgs, val_c2ws = images[val_idxs], c2ws[val_idxs]

        train_idxs = [i for i in range(images.shape[0]) if i not in val_idxs]
        train_imgs, train_c2ws = images[train_idxs], c2ws[train_idxs]

        if stage == "fit":
            self.train_rays: U.data.RayDataset = U.data.RayDataset(
                images=train_imgs,
                c2ws=train_c2ws,
                focal=torch.tensor(self.hparams.focal, dtype=torch.float32),
                rays_per_image=self.hparams.rays_per_image,
                length=self.hparams.epoch_size,
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
            num_workers=1,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_angles,
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )


class LNeRF(L.LightningModule):
    def __init__(self, hidden_size: int = 64, encoding_log2: int = 19, embed_dims: int = 2, levels: int = 16,
                 min_res: int = 16, max_res: int = 2048, max_res_dense: int = 256, f_res: int = 128,
                 f_sigma_init: float = 0.04, f_sigma_threshold: float = 0.01, f_stochastic_test: bool = True,
                 f_update_decay: float = 0.7, f_update_noise_scale: float = None, f_update_selection_rate: float = 0.25,
                 coarse_samples: int = 64, fine_samples: int = 128, **kwargs):
        """Init

        Args:
            hidden_size: Hidden size for Linear layers
            encoding_log2: Log2 of encoding count for MLHHE
            embed_dims: Output embedding dimensions for MLHHE
            levels: Level count for MLHHE
            min_res: Minimal resolution of MLHHE
            max_res: Max resolution of MLHHE
            max_res_dense: Resolution to swap to sparse encoding for MLHHE
            f_res: Occupancy Grid Filter resolution
            f_sigma_init: OGF density init
            f_sigma_threshold: OGF density threshold
            f_stochastic_test: Toggles OGF stochastic test
            f_update_decay: OGF update decay
            f_update_noise_scale: OGF update noise scale
            f_update_selection_rate: Rate of OGF update selection
            coarse_samples: Initial samples to take per ray
            fine_samples: Hierarchical resampling sample count
        """
        super().__init__()
        self.save_hyperparameters()
        self.nerf = U.nn.InstantNGP(
            hidden_size=self.hparams.hidden_size,
            encoding_log2=self.hparams.encoding_log2,
            embed_dims=self.hparams.embed_dims,
            levels=self.hparams.levels,
            min_res=self.hparams.min_res,
            max_res=self.hparams.max_res,
            max_res_dense=self.hparams.max_res_dense,
            f_res=self.hparams.f_res,
            f_sigma_init=self.hparams.f_sigma_init,
            f_sigma_threshold=self.hparams.f_sigma_threshold,
            f_stochastic_test=self.hparams.f_stochastic_test,
            f_update_decay=self.hparams.f_update_decay,
            f_update_noise_scale=self.hparams.f_update_noise_scale,
            f_update_selection_rate=self.hparams.f_update_selection_rate,
        )

        self.lossf = MSELoss(reduction='none')
        self.psnr = PeakSignalNoiseRatio()

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
        points, expanded_directions, coarse_depths = U.rays.sample_ray_uniformally(
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

        points, expanded_directions, fine_depths = U.rays.sample_ray_hierarchically(
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
        origins, directions = U.rays.create_rays(
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
            rgb, _, _ = U.rays.render_rays(rgbs, depths)
            image.append(rgb)

        return torch.cat(image, 0).reshape(height, width, -1)

    def training_step(self, batch, batch_idx):
        pointers, origins, directions, colors = batch
        coarse_rgbs, coarse_depths, fine_rgbs, fine_depths = self.compute_along_rays(origins, directions)

        coarse_colors, _, coarse_alphas = U.rays.render_rays(rgbs=coarse_rgbs, depths=coarse_depths)
        fine_colors, _, fine_alphas = U.rays.render_rays(rgbs=fine_rgbs, depths=fine_depths)
        
        if colors.shape[-1] == 4:  # RGBA, apply background noise to ensure 0 density background
            colors, alpha = colors[..., :3], colors[..., 3:4]
            noise = torch.empty_like(colors).uniform_(0.0, 1.0)

            mixed_colors = colors * alpha + noise * (1 - alpha)
            mixed_coarse_colors = coarse_colors * coarse_alphas + noise * (1 - coarse_alphas)
            mixed_fine_colors = fine_colors * fine_alphas + noise * (1 - fine_alphas)

            loss = (
                self.lossf(mixed_coarse_colors, mixed_colors) + self.lossf(mixed_fine_colors, mixed_colors)
            ).mean(-1)
        else:  # RGB
            loss = (self.lossf(coarse_colors, colors) + self.lossf(fine_colors, colors)).mean(-1)
        self.trainer.datamodule.train_rays.update_errors(pointers, loss)
        loss = loss.mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_imgs = []

    def validation_step(self, batch, batch_idx):
        c2w, focal, image = batch
        image = image[..., :3]  # Limit RGBA to RGB
        render = self.render_image(image.shape[1], image.shape[2], c2w[0], focal[0]).unsqueeze(0)
        loss = self.lossf(render, image).mean()

        render, image = render.permute(0, 3, 1, 2), image.permute(0, 3, 1, 2)
        psnr = self.psnr(render, image)

        metrics = {"val_loss": loss, "val_psnr": psnr}
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        self.val_imgs.append(render)
        return metrics

    def on_validation_epoch_end(self):
        if self.trainer and not self.trainer.sanity_checking:  # Disable image logging on sanity check
            images = torch.cat(self.val_imgs, dim=0)
            self.logger.experiment.add_image("Renders", make_grid(images, nrow=4, padding=5), self.global_step)
        self.val_imgs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.nerf.mlhhe.parameters(), "weight_decay": 0.},
            {"params": self.nerf.rgb_mlp.parameters(), "weight_decay": 10**-6},
            {"params": self.nerf.feature_mlp.parameters(), "weight_decay": 10**-6}
        ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, min_lr=1e-4, factor=0.75, patience=2, mode="max"
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_psnr",
            }
        }


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
            trainer.datamodule.train_rays.update_weights()

            weights = [trainer.datamodule.train_rays.data[i][3].weights for i in list(range(8))]
            weights = torch.stack(weights, dim=0).reshape(-1, 1, 200, 200).expand(-1, 3, -1, -1)
            trainer.logger.experiment.add_image("Sample weights", make_grid(weights, nrow=4, padding=5), trainer.global_step)
        return super().on_validation_epoch_end(trainer, pl_module)


if __name__ == '__main__':
    # Remove warning for train dataloader having num_workers=1
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*Consider setting `persistent_workers*")
    # Remove warning for iterabledataset __len__
    warnings.filterwarnings("ignore", ".*Your `IterableDataset` has `__len__` defined.*")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    accumulation = 2**14
    data = NeRFData("Shurtape_Tape_Purple_CP28", batch_size=2**12, epoch_size=2**21, rays_per_image=2**14)
    module = LNeRF(encoding_log2=14, max_res=512, levels=8, hidden_size=64)
    logger = TensorBoardLogger(".", default_hp_metric=False)

    batches_in_epoch = data.hparams.epoch_size // data.hparams.batch_size
    trainer = L.Trainer(
        max_epochs=100, check_val_every_n_epoch=1,
        log_every_n_steps=1, logger=logger, reload_dataloaders_every_n_epochs=1,
        accumulate_grad_batches=accumulation // data.hparams.batch_size if data.hparams.batch_size < accumulation else 1,
        callbacks=[
            PixelSamplerUpdateCallback(),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(filename="best_val_psnr_{epoch}", monitor="val_psnr", mode="max", every_n_epochs=1),
            ModelCheckpoint(filename="best_train_loss_{step}", monitor="train_loss", mode="min"),
            ModelCheckpoint(filename="{epoch}", every_n_epochs=1),
        ]
    )

    """TODO: 2025.04.11.
    - subpixel sampling by modulating direction (e.g. torch.randn * pixel size/2 in world coords)
    - perhaps custom mlhhe implementation for better understanding, smaller overhead
    - consider pixel sampling weight matrix as NxM, and do some form of blur (e.g. Gaussian) instead of setting to mean
    """

    trainer.fit(model=module, datamodule=data)
