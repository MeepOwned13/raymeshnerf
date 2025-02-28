import torch
from torch import Tensor
from torch.nn import MSELoss, Upsample
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.image import PeakSignalNoiseRatio
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint

import utils as U


class NeRFData(L.LightningDataModule):
    def __init__(self, path: str, batch_size: int = 1024, swap_strategy_iter: int = 10_000,
                 edge_weight_epsilon: float = 0.33, horizontal_val_angles: int = 4,
                 vertical_val_angles: int = 3):
        """Init

        Args:
            path: Path to data file
            batch_size: Batch size and epoch size
            swap_strategy_iter: Specifies at which iteration Squared Error sampling takes over fully
            edge_weight_epsilon: Epsilon to use when deciding pixel weights after edge detection
        """
        super().__init__()
        self.save_hyperparameters()

    def load_from_file(self):
        ext = self.hparams.path.split(".")[-1]
        if ext == "npz":
            return U.data.load_npz(path=self.hparams.path)

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
            origins, directions, colors, pixel_weights = U.data.create_nerf_data(
                images=train_imgs,
                c2ws=train_c2ws,
                focal=torch.tensor(self.hparams.focal, dtype=torch.float32),
                weight_epsilon=self.hparams.edge_weight_epsilon
            )

            self.train_rays: TensorDataset = TensorDataset(torch.arange(origins.shape[0]), origins, directions, colors)
            """Dataset: (i, origins, directions, colors)"""

            self.train_weights: Tensor = pixel_weights
            """Edge weights for training rays"""

            self.val_data: TensorDataset = TensorDataset(
                val_c2ws,
                torch.tensor([focal], dtype=torch.float32).expand(val_c2ws.shape[0]),
                val_imgs
            )
            """Dataset: (c2w, focal, image)"""

    def train_dataloader(self):
        self.train_sampler: U.data.ImportantPixelSampler = U.data.ImportantPixelSampler(
            weights=self.train_weights,
            num_samples=self.hparams.batch_size,
            replacement=True,
            swap_strategy_iter=self.hparams.swap_strategy_iter
        )
        return DataLoader(
            dataset=self.train_rays,
            batch_size=self.hparams.batch_size,
            sampler=self.train_sampler,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )


class LNeRF(L.LightningModule):
    def __init__(self, num_layers: int = 8, hidden_size: int = 256, in_coordinates: int = 3, in_directions: int = 3,
                 skips: list[int] = [4], coord_encode_freq: int = 10, dir_encode_freq: int = 4,
                 coarse_samples: int = 64, fine_samples: int = 64, lr: float = 5e-5, **kwargs):
        """Init

        Args:
            num_layers: Layer count for primary feature MLP
            hidden_size: Hidden size for all Linear layers
            in_coordinates: Count of input point coordinates
            in_directions: Count of input direction coordinates (spherical=>2, cartesian=>3)
            skips: Skip connection list for primary feature MLP
            coord_encode_freq: Max frequency for coordinate PE
            dir_encode_freq: Max frequency for direction PE
            lr: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()
        self.nerf = U.nn.NeRF(
            num_layers=self.hparams.num_layers,
            hidden_size=self.hparams.hidden_size,
            in_coordinates=self.hparams.in_coordinates,
            in_directions=self.hparams.in_directions,
            skips=self.hparams.skips,
            coord_encode_freq=self.hparams.coord_encode_freq,
            dir_encode_freq=self.hparams.dir_encode_freq,
        )

        self.mse = MSELoss(reduction='none')
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
            rgb, _ = U.rays.render_rays(rgbs, depths)
            image.append(rgb)

        return torch.cat(image, 0).reshape(height, width, -1)

    def training_step(self, batch, batch_idx):
        i, origins, directions, colors = batch
        coarse_rgbs, coarse_depths, fine_rgbs, fine_depths = self.compute_along_rays(origins, directions)

        coarse_colors, _ = U.rays.render_rays(rgbs=coarse_rgbs, depths=coarse_depths)
        fine_colors, _ = U.rays.render_rays(rgbs=fine_rgbs, depths=fine_depths)
        loss = (self.mse(coarse_colors, colors) + self.mse(fine_colors, colors)).mean(dim=-1)
        # Updating squared errors in ImportantPixelSampler to re-weight the used rays
        self.trainer.train_dataloader.sampler.update_errors(idxs=i.cpu(), errors=loss)

        loss = loss.mean()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_start(self):
        self.val_imgs = []

    def validation_step(self, batch, batch_idx):
        c2w, focal, image = batch
        render = self.render_image(image.shape[1], image.shape[2], c2w[0], focal[0]).unsqueeze(0)
        loss = self.mse(render, image).mean()

        render, image = render.permute(0, 3, 1, 2), image.permute(0, 3, 1, 2)
        psnr = self.psnr(render, image)

        metrics = {"val_loss": loss, "val_psnr": psnr}
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

        self.val_imgs.append(render)
        return metrics

    def on_validation_epoch_end(self):
        if self.current_epoch != 0:
            images = torch.cat(self.val_imgs, dim=0)
            with torch.no_grad():
                images = Upsample(scale_factor=2, mode="nearest")(images)  # upsampling for better display
            self.logger.experiment.add_image("Renders", make_grid(images, nrow=4, padding=5), self.current_epoch)

        self.val_imgs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.nerf.parameters(), lr=self.hparams.lr)


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    data = NeRFData("data/tiny_nerf_data.npz")
    module = LNeRF()
    logger = TensorBoardLogger(".", default_hp_metric=False)
    # TODO: make progress bar display progress of epochs
    trainer = L.Trainer(
        max_epochs=100_001, check_val_every_n_epoch=1000, log_every_n_steps=1, logger=logger,
        callbacks=[
            ModelCheckpoint(filename="best_val_psnr_{epoch}", monitor="val_psnr", mode="max"),
            ModelCheckpoint(filename="{epoch}", every_n_epochs=200, monitor="epoch",
                            mode="max", save_on_train_epoch_end=True),
            RichProgressBar(),
        ]
    )
    trainer.fit(module, datamodule=data)

