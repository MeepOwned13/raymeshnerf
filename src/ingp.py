import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import utils as U
import utils.lutils as LU


class LInstantNGP(LU.LVolume):
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
        super().__init__(coarse_samples=coarse_samples, fine_samples=fine_samples)
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


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    acc = 2**4
    data = LU.NeRFData(
        "checkered_cube", batch_size=2**11, epoch_size=2**22, rays_per_image=2**11,
    )
    module = LInstantNGP()
    logger = TensorBoardLogger(".", default_hp_metric=False)

    batches_in_epoch = data.hparams.epoch_size // data.hparams.batch_size
    trainer = L.Trainer(
        max_epochs=200, check_val_every_n_epoch=1, log_every_n_steps=1, logger=logger,
        accumulate_grad_batches=acc,
        callbacks=[
            LU.OGFilterCallback(2**22 // data.hparams.batch_size // acc),
            LU.PixelSamplerUpdateCallback(),
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(filename="best_val_psnr_{epoch}", monitor="val_psnr", mode="max", every_n_epochs=1),
            ModelCheckpoint(filename="best_train_loss_{step}", monitor="train_loss", mode="min"),
            ModelCheckpoint(filename="{epoch}", every_n_epochs=1),
        ],
    )

    trainer.fit(model=module, datamodule=data)
