import torch
from torch import Tensor
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import utils as U
import utils.lutils as LU


class LNeRF(LU.LVolume):
    def __init__(self, num_layers: int = 8, hidden_size: int = 256, in_coordinates: int = 3, in_directions: int = 3,
                 skips: list[int] = [4], coord_encode_freq: int = 10, dir_encode_freq: int = 4,
                 coarse_samples: int = 64, fine_samples: int = 128, lr: float = 1e-4,
                 weight_decay: float = 1e-8, **kwargs):
        """Init

        Args:
            num_layers: Layer count for primary feature MLP
            hidden_size: Hidden size for all Linear layers
            in_coordinates: Count of input point coordinates
            in_directions: Count of input direction coordinates (spherical=>2, cartesian=>3)
            skips: Skip connection list for primary feature MLP
            coord_encode_freq: Max frequency for coordinate PE
            dir_encode_freq: Max frequency for direction PE
            coarse_samples: Initial samples to take per ray
            fine_samples: Hierarchical resampling sample count
            lr: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()
        self.nerf: U.nn.NeRF = U.nn.NeRF(
            num_layers=self.hparams.num_layers,
            hidden_size=self.hparams.hidden_size,
            in_coordinates=self.hparams.in_coordinates,
            in_directions=self.hparams.in_directions,
            skips=self.hparams.skips,
            coord_encode_freq=self.hparams.coord_encode_freq,
            dir_encode_freq=self.hparams.dir_encode_freq,
        )

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
            tuple: tuple containing (coarse_rgbs, coarse_depths, fine_rgbs, fine_depths)
            - **coarse_rgbs**: *shape[N, coarse_samples, 4]*: RGBS predicted by NeRF for uniform samples
            - **coarse_depths**: *shape[N, coarse_samples]*: Depths sampled uniformally, sorted and aligned to coarse_rgbs
            - **fine_rgbs**: *shape[N, fine_samples, 4]*: RGBS predicted by NeRF for hierarchical samples
            - **fine_depths**: *shape[N, fine_samples]*: Depths sampled hierarchically, sorted and aligned to fine_rgbs
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

    def render_rays(self, origins, directions, near, far):
        near = self.hparams.get("near", near) or self.trainer.datamodule.hparams.near
        far = self.hparams.get("far", far) or self.trainer.datamodule.hparams.far

        _, _, rgbs, depths = self.compute_along_rays(origins, directions, near, far)
        rgb, depth, acc, _, _ = U.rays.render_rays(rgbs=rgbs, depths=depths, far=far)
        return rgb, depth, acc
    
    def calculate_loss(self, origins, directions, colors):
        far = self.hparams.get("far", None) or self.trainer.datamodule.hparams.far
        coarse_rgbs, coarse_depths, fine_rgbs, fine_depths = self.compute_along_rays(origins, directions)

        coarse_colors, _, coarse_alphas, _, _ = U.rays.render_rays(
            rgbs=coarse_rgbs, depths=coarse_depths, far=far
        )
        fine_colors, _, fine_alphas, _, _ = U.rays.render_rays(
            rgbs=fine_rgbs, depths=fine_depths, far=far
        )

        if colors.shape[-1] == 4:  # RGBA, apply background noise to skew towards low density background
            colors, alphas = colors[..., :3], colors[..., 3:4]
            noise = torch.empty_like(colors).uniform_(self.background_noise_range[0], self.background_noise_range[1])

            mixed_colors = colors * alphas + noise * (1 - alphas)
            mixed_coarse_colors = coarse_colors * coarse_alphas + noise * (1 - coarse_alphas)
            mixed_fine_colors = fine_colors * fine_alphas + noise * (1 - fine_alphas)

            loss = (
                self.lossf(mixed_coarse_colors, mixed_colors) + self.lossf(mixed_fine_colors, mixed_colors)
            )
        else:  # RGB
            loss = (self.lossf(coarse_colors, colors) + self.lossf(fine_colors, colors))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.nerf.parameters(), lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, min_lr=1e-6, factor=0.7, patience=2, mode="max", cooldown=1
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_psnr",
            }
        }


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    L.seed_everything(42)
    decay = 5e-7

    data = LU.NeRFData("Shurtape_Tape_Purple_CP28", batch_size=2**9)
    module = LNeRF(weight_decay=decay, coarse_samples=128)
    logger = TensorBoardLogger(".", default_hp_metric=False, version=f"shurtape200x200_decay={decay:.0e}")

    batches_in_epoch = data.hparams.epoch_size // data.hparams.batch_size
    trainer = L.Trainer(
        max_epochs=15, check_val_every_n_epoch=1, log_every_n_steps=1, logger=logger,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(filename="best_val_psnr_{epoch}", monitor="val_psnr", mode="max", every_n_epochs=1,
                            save_weights_only=True),
            ModelCheckpoint(filename="end_{epoch}", save_on_train_epoch_end=True, every_n_epochs=1),
        ]
    )

    trainer.fit(
        model=module, datamodule=data,
        ckpt_path=None
    )
