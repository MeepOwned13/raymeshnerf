"""
Based on code from pure-torch-ngp (MIT License): https://github.com/cheind/pure-torch-ngp

Modifications:
- Removed stochastic tests and changing of update noise scale
- Grid can now use full tests (suggested for early stages of training, see utils/lutils->OGFilterCallback)
- Selection of updates align with https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf E.2, as in half
  of the gridpoints are selected, half of that using random uniform samples and the others using rejection sampling
  (taking only "dense" areas for sampling)
- Default parameters were adjusted to match the mentioned paper and the [-1, 1] bounding box used for the repo
"""

import torch


def make_grid(
    shape: tuple[int, ...],
    indexing: str = "xy",
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.LongTensor:
    """Generalized mesh-grid routine.

    torch.meshgrid `indexing='xy'` only works for 2 dimensions and switches to 'ij'
    for more than two dimensions. This method is consistent for any number of dims.

    Params:
        shape: shape of grid to generate coordinates for
        indexing: order of coordinates in last-dimension
        device: device to put it on
        dtype: dtype to return

    Returns:
        coords: (shape,)+(dims,) tensor
    """
    ranges = [torch.arange(r, device=device, dtype=dtype) for r in shape]
    coords = torch.stack(torch.meshgrid(*ranges, indexing="ij"), -1)
    if indexing == "xy":
        coords = torch.index_select(
            coords, -1, torch.arange(len(shape), device=device).flip(0)
        )
    return coords


class OccupancyGridFilter(torch.nn.Module):
    def __init__(
        self,
        res: int = 128,
        density_initial: float = 5.0,
        density_threshold: float = 2.956033378,
        update_decay: float = 0.95,
        update_selection_rate: float = 0.5,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.res = res
        self.update_decay = update_decay
        self.density_initial = density_initial
        self.density_threshold = density_threshold
        self.update_selection_rate = update_selection_rate
        self.density_grid = torch.nn.Buffer(torch.full((res, res, res), density_initial))
        self.bool_grid = torch.nn.Buffer(torch.full((res, res, res), True))

    def test(self, xyz_ndc: torch.Tensor) -> torch.BoolTensor:
        mask = ((xyz_ndc >= -1.0) & (xyz_ndc <= 1.0)).all(-1)

        ijk = (xyz_ndc + 1) * self.res * 0.5 - 0.5
        ijk = torch.round(ijk).clamp(0, self.res - 1).long()
        d_mask = self.bool_grid[ijk[..., 2], ijk[..., 1], ijk[..., 0]]
        return mask & d_mask

    @torch.no_grad()
    def update(self, nerf, full_selection: bool = False):
        self.density_grid *= self.update_decay
        kernel_lim = 2**19

        if full_selection or self.update_selection_rate >= 1.0:
            ijk = make_grid(
                (self.res, self.res, self.res),
                indexing="xy",
                device=self.density_grid.device,
                dtype=torch.long,
            ).view(-1, 3)
        else:
            # Half as uniform random, half as rejection sample (only ones with True in bool grid)
            M = int(self.update_selection_rate * self.res**3)
            ijk = torch.zeros((M, 3), dtype=int, device=self.density_grid.device)

            ijk[0:M//2] = torch.randint(0, self.res, size=(M // 2, 3), device=self.density_grid.device)
            rej = torch.argwhere(self.bool_grid)

            # Chunking to stay inside kernel_lim (masking would fail)
            mask_chunks = torch.linspace(0, rej.shape[0], int(torch.ceil(torch.tensor(rej.shape[0] / kernel_lim + 1))))
            ijk_chunks = torch.linspace(0, M // 2, mask_chunks.shape[0])
            for i in range(1, mask_chunks.shape[0]):
                mask_start, mask_end = int(torch.floor(mask_chunks[i-1])), int(torch.ceil(mask_chunks[i]))
                ijk_start, ijk_end = int(torch.floor(ijk_chunks[i-1])), int(torch.ceil(ijk_chunks[i]))

                rands = torch.randint(0, mask_end - mask_start, (ijk_end - ijk_start,))
                ijk[M//2:][ijk_start:ijk_end] = rej[mask_start:mask_end][rands]

        noise = torch.rand_like(ijk, dtype=torch.float) - 0.5
        xyz = ijk + noise
        xyz_ndc = (xyz + 0.5) * 2 / self.res - 1.0

        # Chunking to 2**19s to maximize kernel usage without going over limit
        d = torch.full((xyz_ndc.shape[0],), torch.nan, dtype=torch.float32, device=self.density_grid.device)
        chunks = torch.arange(0, xyz_ndc.shape[0], kernel_lim)
        for c in chunks:
            d[c:c+kernel_lim] = nerf(
                xyz_ndc[c:c+kernel_lim], directions=None, skip_colors=True, masked=False
            ).squeeze(-1)

        cur = self.density_grid[ijk[:, 2], ijk[:, 1], ijk[:, 0]]
        new = torch.maximum(d, cur)
        self.density_grid[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = new
        self.bool_grid[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = new > self.density_threshold
