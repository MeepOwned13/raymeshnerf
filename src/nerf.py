import torch
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
        super().__init__(coarse_samples=coarse_samples, fine_samples=fine_samples)
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.nerf.parameters(), lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, min_lr=1e-6, factor=0.7, patience=2, mode="max", cooldown=2
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
    decay = 1e-6

    data = LU.NeRFData(
        "Weisshai_Great_White_Shark", batch_size=2**9, epoch_size=2**20, rays_per_image=2**10
    )
    module = LNeRF(weight_decay=decay)
    logger = TensorBoardLogger(".", default_hp_metric=False, version=f"weisshai_shark200x200_decay={decay:.0e}")

    batches_in_epoch = data.hparams.epoch_size // data.hparams.batch_size
    trainer = L.Trainer(
        max_epochs=40, check_val_every_n_epoch=1, log_every_n_steps=1, logger=logger,
        callbacks=[
            LU.PixelSamplerUpdateCallback(),
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(filename="best_val_psnr_{epoch}", monitor="val_psnr", mode="max", every_n_epochs=1,
                            save_weights_only=True),
            ModelCheckpoint(filename="end_{epoch}", save_on_train_epoch_end=True, every_n_epochs=1),
        ],
    )

    trainer.fit(
        model=module, datamodule=data,
        ckpt_path=None
    )
