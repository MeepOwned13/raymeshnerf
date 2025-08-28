import torch
from torch.utils.checkpoint import checkpoint
from torch import Tensor
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import utils as U
import utils.lutils as LU

from utils.eff_distloss import eff_distloss

class LInstantNGP(LU.LVolume):
    def __init__(self, hidden_size: int = 64, encoding_log2: int = 19, embed_dims: int = 2, levels: int = 16,
                 min_res: int = 16, max_res: int = 2048, max_res_dense: int = 256, f_res: int = 128,
                 f_sigma_init: float = 5.0, f_sigma_threshold: float = 2.956033378, f_update_decay: float = 0.95,
                 f_update_selection_rate: float = 0.5, dl_tv_loss_decay_end: int = 2**13 + 2**12,
                 distortion_loss_weight_start: float = 1e-4, distortion_loss_weight_end: float = 1e-2,
                 tv_loss_weight_start: float = 1e-6, tv_loss_weight_end: float = 1e-8, 
                 ray_tv_sample_count: int = 2 ** 8, ray_tv_loss_mult: float = 20.0, **kwargs):
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
            dl_tv_loss_decay_end: Linear decay ending batch number for distortion and tv loss,
                tv loss isn't applied on subsequent batches
            distortion_loss_weight_start: Starting Distortion loss weight
            distortion_loss_weight_end: Ending Distortion loss weight, scaled linearly even after `dl_tv_loss_decay_end`
            tv_loss_weight_start: Starting Total Variation loss weight
            tv_loss_weight_end: Ending Total Variation loss weight at `dl_tv_loss_decay_end`
            ray_tv_sample_count: Count of sample per ray for additional TV loss
            ray_tv_loss_mult: Scale tv loss weight by this factor for ray tv loss
        """
        super().__init__()
        if (ray_tv_sample_count > 2 ** 10 and ray_tv_sample_count % 2 == 0) or ray_tv_sample_count < 0:
            raise ValueError("ray_tv_sample_count must be divisible by 2 and remain between 0 and 1024 (total samples per ray)")

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
            perturb=not deterministic,
        )
        rgbs = self.nerf(points, expanded_directions)
        rgb, depth, acc, _, _ = U.rays.render_rays(rgbs=rgbs, depths=depths, far=far)
        return rgb, depth, acc
    
    def calculate_loss(self, origins, directions, colors):
        near = self.hparams.get("near", self.trainer.datamodule.hparams.near)
        far = self.hparams.get("far", self.trainer.datamodule.hparams.far)
        batch_size = origins.shape[0]

        points, expanded_directions, depths = U.rays.sample_ray_uniformally(
            origins=origins,
            directions=directions,
            near=near,
            far=far,
            num_samples=2**10,
            perturb=True,
        )
        p_rgbs = self.nerf(points, expanded_directions)
        p_rgb, _, p_alpha, _, p_weights = U.rays.render_rays(rgbs=p_rgbs, depths=depths, far=far)

        # Distortion loss
        distances = depths[..., 1:] - depths[..., :-1]
        distances = torch.cat([distances, torch.nn.functional.relu(far - depths[..., -1:])], -1)
        distloss = eff_distloss(p_weights, depths, distances)

        if colors.shape[-1] == 4:  # RGBA, apply background noise to skew towards low density background
            colors, alphas = colors[..., :3], colors[..., 3:4]
            noise = torch.empty_like(colors).uniform_(self.background_noise_range[0], self.background_noise_range[1])

            mixed_colors = colors * alphas + noise * (1 - alphas)
            mixed_pred_colors = p_rgb * p_alpha + noise * (1 - p_alpha)

            loss = self.lossf(mixed_pred_colors, mixed_colors)
        else:  # RGB
            loss = self.lossf(p_rgb * p_alpha, colors)

        # Distortion loss
        ds_loss_weight = self.hparams.distortion_loss_weight_start -\
            self.trainer.global_step / self.hparams.dl_tv_loss_decay_end *\
            (self.hparams.distortion_loss_weight_start - self.hparams.distortion_loss_weight_end)
        loss += distloss * ds_loss_weight

        if self.trainer.global_step <= self.hparams.dl_tv_loss_decay_end:
            tv_loss_weight = self.hparams.tv_loss_weight_start -\
                self.trainer.global_step / self.hparams.dl_tv_loss_decay_end *\
                (self.hparams.tv_loss_weight_start - self.hparams.tv_loss_weight_end)
            loss += self.nerf.mlhhe.tv_loss(batch_size * 2) * tv_loss_weight

            if self.hparams.ray_tv_sample_count > 0:
                # Total Variation applied to subset of points centered on max sigma of ray
                half_sample_count = self.hparams.ray_tv_sample_count // 2
                ch_idxs = torch.argmax(p_weights, dim=-1, keepdim=True)
                ch_idxs = ch_idxs.clamp(half_sample_count, 2**10 - half_sample_count)
                ch_idxs = ch_idxs + torch.arange(-half_sample_count, half_sample_count, device=self.device)
                row_indices = torch.arange(
                    batch_size, device=self.device
                ).unsqueeze(1).expand(-1, self.hparams.ray_tv_sample_count)
                ch_rgbs = p_rgbs[row_indices, ch_idxs, :]
                ch_points, ch_dirs = points[row_indices, ch_idxs, :], expanded_directions[row_indices, ch_idxs, :]

                # Offset with random perpendicular (to each other as well) vectors
                tangent = torch.cross(
                    ch_dirs, ch_dirs[..., [2, 0, 1]] * torch.tensor([-1, 1, 1], device=self.device), dim=-1
                )
                bitangent = torch.cross(ch_dirs, tangent, dim=-1)
                angle = ((torch.rand(ch_dirs.shape[0], device=self.device) * 2 - 1) * torch.pi)[:, None, None]
                a = torch.nn.functional.normalize(tangent * torch.sin(angle) + bitangent * torch.cos(angle), dim=-1)
                b = torch.cross(ch_dirs, a, dim=-1)

                # Offsetting by (2 * sqrt(3) / 1024) = 0.0033829116728156805 (~step_size)
                offset = 0.0033829116728156805
                if self.hparams.ray_tv_sample_count < 2 ** 9:
                    # Catting the x and y shift to make the calculation a single call
                    a_rgbs, b_rgbs = self.nerf(
                        torch.cat([ch_points + a * offset, ch_points + b * offset], dim=0),
                        torch.cat([ch_dirs, ch_dirs], dim=0)
                    ).split(batch_size, dim=0)
                else:  # Needs to be 2 calls to stay under 2^19 kernel size limit (2**9 batch size * sample_count)
                    a_rgbs= self.nerf(ch_points + a * offset, ch_dirs)
                    b_rgbs= self.nerf(ch_points + b * offset, ch_dirs)

                rtv = torch.abs(ch_rgbs - a_rgbs) + torch.abs(ch_rgbs - b_rgbs)
                rtv = rtv.sum(-1).sum(-1).mean()
                loss += rtv * tv_loss_weight * self.hparams.ray_tv_loss_mult

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
                    optimizer, gamma=0.9
                ),
            }
        }


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    L.seed_everything(42, workers=True)

    data = LU.NeRFData("scan24", U.data.ObjectSource.DTU, batch_size=2**9, val_angle_indices=[1, -16], keep_val_in_train=True)
    module = LInstantNGP()
    logger = TensorBoardLogger(".", default_hp_metric=False, version=f"ingp_{data.scene_name}_m")

    trainer = L.Trainer(
        max_epochs=15, check_val_every_n_epoch=1, log_every_n_steps=1, logger=logger,
        accumulate_grad_batches=2**2, limit_train_batches=2**12 * 2**2,
        callbacks=[
            LU.OGFilterCallback(16 * 2**4, 32),
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(filename="best_val_psnr_{epoch}", monitor="val_psnr", mode="max", every_n_epochs=1,
                            save_weights_only=True),
            ModelCheckpoint(filename="end_{epoch}", save_on_train_epoch_end=True, every_n_epochs=1),
            EarlyStopping(monitor="val_psnr", mode="max", patience=2, min_delta=0.05)
        ],
        num_sanity_val_steps=0,  # Here to reduce experiment time, make sure to set it to -1 or >0 for new data(sets)
    )

    trainer.fit(
        model=module, datamodule=data
    )
