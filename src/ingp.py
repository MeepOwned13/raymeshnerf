import torch
from torch.utils.checkpoint import checkpoint
from torch import Tensor
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import utils as U
import utils.lutils as LU


class LInstantNGP(LU.LVolume):
    def __init__(self, hidden_size: int = 64, encoding_log2: int = 19, embed_dims: int = 2, levels: int = 16,
                 min_res: int = 16, max_res: int = 512, max_res_dense: int = 256, f_res: int = 128,
                 f_sigma_init: float = 5.0, f_sigma_threshold: float = 2.956033378,
                 f_update_decay: float = 0.95, f_update_selection_rate: float = 0.5,
                 **kwargs):
        """Init

        Default f_sigma_threshold is chosen based on https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf,
        they state that threshold is 0.01 * step_size, in this case step size is double as in the paper as we use
        the [-1,1] bounding box instead of [0,1]

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
            f_sigma_threshold: OGF density threshold, defaults to 0.01 * 1024 / (2 * sqrt(3))
            f_update_decay: OGF update decay
            f_update_selection_rate: Rate of OGF update selection
        """
        super().__init__()
        self.save_hyperparameters()
        self.nerf: U.nn.InstantNGP = U.nn.InstantNGP(
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
            f_update_decay=self.hparams.f_update_decay,
            f_update_selection_rate=self.hparams.f_update_selection_rate,
        )

        self.background_noise_range = [0.0, 1.0]

    def render_rays(self, origins: Tensor, directions: Tensor, near: float | None = None,
                           far: float | None = None, deterministic: bool = True,
                            **kwargs) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        near = self.hparams.get("near", near) or self.trainer.datamodule.hparams.near
        far = self.hparams.get("far", far) or self.trainer.datamodule.hparams.far

        points, expanded_directions, depths = U.rays.sample_ray_uniformally(
            origins=origins,
            directions=directions,
            near=near,
            far=far,
            num_samples=2**10,
        )
        rgbs = self.nerf(points, expanded_directions)
        rgb, depth, acc, _, _ = U.rays.render_rays(rgbs=rgbs, depths=depths, far=far)
        return rgb, depth, acc
    
    def calculate_loss(self, origins, directions, colors):
        p_rgb, _, p_alpha = self.render_rays(origins, directions)

        if colors.shape[-1] == 4:  # RGBA, apply background noise to skew towards low density background
            colors, alphas = colors[..., :3], colors[..., 3:4]
            noise = torch.empty_like(colors).uniform_(self.background_noise_range[0], self.background_noise_range[1])

            mixed_colors = colors * alphas + noise * (1 - alphas)
            mixed_pred_colors = p_rgb * p_alpha + noise * (1 - p_alpha)

            loss = self.lossf(mixed_pred_colors, mixed_colors).mean(-1)
        else:  # RGB
            loss = self.lossf(p_rgb, colors).mean(-1)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.RAdam([
            {"params": self.nerf.mlhhe.parameters(), "weight_decay": 0.},
            {"params": self.nerf.rgb_mlp.parameters(), "weight_decay": 10**-6, "eps": 1e-15},
            {"params": self.nerf.feature_mlp.parameters(), "weight_decay": 10**-6, "eps": 1e-15}
        ], lr=1e-2, betas=(0.9, 0.99))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.6
                ),
            }
        }


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    L.seed_everything(42)

    data = LU.NeRFData(
        "Weisshai_Great_White_Shark", batch_size=2**9, epoch_size=2**20, rays_per_image=2**8,
    )
    module = LInstantNGP()
    logger = TensorBoardLogger(".", default_hp_metric=False, version=f"ingp_weisshai_shark400x400")

    batches_in_epoch = data.hparams.epoch_size // data.hparams.batch_size
    trainer = L.Trainer(
        max_epochs=20, check_val_every_n_epoch=1, log_every_n_steps=1, logger=logger,
        accumulate_grad_batches=2**4,
        callbacks=[
            LU.OGFilterCallback(16, 8),
            LU.PixelSamplerUpdateCallback(32),
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(filename="best_val_psnr_{epoch}", monitor="val_psnr", mode="max", every_n_epochs=1,
                            save_weights_only=True),
            ModelCheckpoint(filename="end_{epoch}", save_on_train_epoch_end=True, every_n_epochs=1),
            EarlyStopping(monitor="val_psnr", mode="max", patience=1, min_delta=0.1)
        ],
        plugins=[
            LU.RemoveCheckpointKeyBasedOnPathCheckpointPlugin("val_psnr", "NeRFData")
        ]
    )

    trainer.fit(
        model=module, datamodule=data
    )
