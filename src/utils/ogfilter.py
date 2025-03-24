"""
MIT License

Copyright (c) 2022 Christoph Heindl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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
        res: int = 64,
        density_initial=0.02,
        density_threshold=0.01,
        stochastic_test: bool = True,
        update_decay: float = 0.7,
        update_noise_scale: float = None,
        update_selection_rate=0.25,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.res = res
        self.update_decay = update_decay
        self.density_initial = density_initial
        self.density_threshold = density_threshold
        self.update_selection_rate = update_selection_rate
        self.stochastic_test = stochastic_test
        if update_noise_scale is None:
            update_noise_scale = 0.9
        self.update_noise_scale = update_noise_scale
        self.register_buffer("grid", torch.full((res, res, res), density_initial))
        self.grid: torch.Tensor

    def test(self, xyz_ndc: torch.Tensor) -> torch.BoolTensor:
        mask = ((xyz_ndc >= -1.0) & (xyz_ndc <= 1.0)).all(-1)

        ijk = (xyz_ndc + 1) * self.res * 0.5 - 0.5
        ijk = torch.round(ijk).clamp(0, self.res - 1).long()

        d = self.grid[ijk[..., 2], ijk[..., 1], ijk[..., 0]]
        d_mask = d > self.density_threshold
        if self.stochastic_test:
            d_stoch_mask = torch.bernoulli(1 - (-(d + 1e-4)).exp()).bool()
            d_mask |= d_stoch_mask

        return mask & d_mask

    @torch.no_grad()
    def update(self, nerf):
        self.grid *= self.update_decay

        if self.update_selection_rate < 1.0:
            M = int(self.update_selection_rate * self.res**3)
            ijk = torch.randint(0, self.res, size=(M, 3), device=self.grid.device)
        else:
            ijk = make_grid(
                (self.res, self.res, self.res),
                indexing="xy",
                device=self.grid.device,
                dtype=torch.long,
            ).view(-1, 3)

        noise = torch.rand_like(ijk, dtype=torch.float) - 0.5
        noise *= self.update_noise_scale
        xyz = ijk + noise
        xyz_ndc = (xyz + 0.5) * 2 / self.res - 1.0

        d = nerf(xyz_ndc, directions=None, skip_colors=True, masked=False)
        cur = self.grid[ijk[:, 2], ijk[:, 1], ijk[:, 0]]
        new = torch.maximum(d, cur)
        self.grid[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = new
