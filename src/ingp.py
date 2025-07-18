import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import utils as U
import utils.lutils as LU


class LInstantNGP(LU.LVolume):
    def __init__(self, hidden_size: int = 64, encoding_log2: int = 19, embed_dims: int = 2, levels: int = 16,
                 min_res: int = 16, max_res: int = 512, max_res_dense: int = 256, f_res: int = 128,
                 f_sigma_init: float = 0.04, f_sigma_threshold: float = 0.01, f_stochastic_test: bool = True,
                 f_update_decay: float = 0.7, f_update_noise_scale: float = None, f_update_selection_rate: float = 0.25,
                 coarse_samples: int = 128, fine_samples: int = 128, **kwargs):
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
            f_stochastic_test=self.hparams.f_stochastic_test,
            f_update_decay=self.hparams.f_update_decay,
            f_update_noise_scale=self.hparams.f_update_noise_scale,
            f_update_selection_rate=self.hparams.f_update_selection_rate,
        )

        self.background_noise_range = [0.0, 1.0]

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
        "Weisshai_Great_White_Shark", batch_size=2**12, epoch_size=2**20, rays_per_image=2**9,
    )
    module = LInstantNGP()
    logger = TensorBoardLogger(".", default_hp_metric=False, version=f"ingp_weisshai_shark400x400")

    batches_in_epoch = data.hparams.epoch_size // data.hparams.batch_size
    trainer = L.Trainer(
        max_epochs=20, check_val_every_n_epoch=1, log_every_n_steps=1, logger=logger,
        callbacks=[
            LU.OGFilterCallback(16),
            LU.PixelSamplerUpdateCallback(64),
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(filename="best_val_psnr_{epoch}", monitor="val_psnr", mode="max", every_n_epochs=1,
                            save_weights_only=True),
            ModelCheckpoint(filename="end_{epoch}", save_on_train_epoch_end=True, every_n_epochs=1),
            EarlyStopping(monitor="val_psnr", mode="max", patience=1, min_delta=0.05)
        ],
        plugins=[
            LU.RemoveCheckpointKeyBasedOnPathCheckpointPlugin("val_psnr", "NeRFData")
        ]
    )

    trainer.fit(
        model=module, datamodule=data
    )
