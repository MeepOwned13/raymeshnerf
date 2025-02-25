import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Appended Positional Encoding Module"""

    def __init__(self, max_freq: int):
        """Init

        Args:
            max_freq: Maximum frequency to use for encoding
        """
        super(PositionalEncoding, self).__init__()
        self._max_freq = max_freq
        """Stores the max frequency"""

        freq_bands = 2.0 ** torch.linspace(0.0, max_freq - 1, steps=max_freq, dtype=torch.float32)
        self._freq_bands = nn.parameter.Buffer(freq_bands)
        """Pre-calculated frequency bands for encoding"""

    def forward(self, x: Tensor) -> Tensor:
        """Perform Positional Encoding

        Args:
            x (shape[..., M]): Tensor to encode

        Returns:
            pe (shape[..., M + M * max_freq * 2]): Tensor with appended Positional Encoding
        """
        encs = (x[..., None] * self._freq_bands).flatten(-2, -1)
        # Encoding to (x, sin parts, cos parts) of shape(N, M+M*max_freq*2) if x is of shape(N,M)
        return torch.cat([x, encs.sin(), encs.cos()], dim=-1)

    def get_out_dim(self, in_dim: int):
        """Returns the outgoing dimension of the forward() method

        Args:
            in_dim: Last dim of Tensor to be used

        Returns:
            out_dim: Last dim after Positional Encoding
        """
        return in_dim + in_dim * self._max_freq * 2


class NeRF(nn.Module):
    """NeRF"""

    def __init__(self, num_layers: int = 8, hidden_size: int = 256, in_coordinates: int = 3, in_directions: int = 3,
                 skips: list[int] = [4], coord_encode_freq: int = 10, dir_encode_freq: int = 4):
        """Init

        Args:
            num_layers: Layer count for primary feature MLP
            hidden_size: Hidden size for all Linear layers
            in_coordinates: Count of input point coordinates
            in_directions: Count of input direction coordinates (spherical=>2, cartesian=>3)
            skips: Skip connection list for primary feature MLP
            coord_encode_freq: Max frequency for coordinate PE
            dir_encode_freq: Max frequency for direction PE
        """
        super(NeRF, self).__init__()
        self.in_coordinates = in_coordinates
        """Count of input point coordinates"""

        self.in_directions = in_directions
        """Count of input direction coordinates (spherical=>2, cartesian=>3)"""

        self.skips = tuple(skips)
        """Skip connection tuple for primary feature MLP"""

        self.coordinate_encoder = PositionalEncoding(coord_encode_freq)
        """Coordinate PE"""

        self.direction_encoder = PositionalEncoding(dir_encode_freq)
        """Direction PE"""

        coord_dim = self.coordinate_encoder.get_out_dim(self.in_coordinates)
        self.feature_mlp = nn.ModuleList([nn.Linear(coord_dim, hidden_size)])
        """Primary feature MLP"""
        # go until num_layers -1 as we already have the initial layer
        for i in range(num_layers - 1):
            self.feature_mlp.append(nn.Sequential(
                # skip with +1 as we already have the initial layer
                nn.Linear(hidden_size + (coord_dim if i + 1 in self.skips else 0), hidden_size),
                nn.ReLU(inplace=True),
            ))

        self.sigma_fc = nn.Linear(hidden_size, 1)
        """Linear layer for sigma calculation"""

        self.color_preproc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        """Linear layer preprocessing features before feeding in direction"""

        dir_dim = self.direction_encoder.get_out_dim(self.in_directions)
        self.rgb_mlp = nn.Sequential(
            nn.Linear(hidden_size + dir_dim, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, 3),
            nn.Sigmoid(),
        )
        """Linear layer for RGB calculation"""

    def forward(self, coordinates: Tensor, directions: Tensor, skip_colors: bool = False) -> Tensor:
        """Perform RGBS calculation

        coordinates and directions must have the same dimensions for *...

        Args:
            coordinates (shape[*..., in_coordinates]): Input point coordinates
            directions (shape[*..., in_directions]): Input directions
            skip_colors: Skip color calculation?

        Returns:
            rgbs (shape[*..., 4]): RGB&Sigma if skip_colors=False
            rgbs (shape[*...]): Sigma if skip_colors=True
        """
        coordinates = self.coordinate_encoder(coordinates)
        features = coordinates
        for i, fc in enumerate(self.feature_mlp):
            if i in self.skips:
                features = torch.cat([features, coordinates], -1)
            features = fc(features)

        sigma = self.sigma_fc(features)

        if skip_colors:
            return sigma

        directions = self.direction_encoder(directions)
        features = self.color_preproc(features)
        features = torch.cat([features, directions], dim=-1)
        rgb = self.rgb_mlp(features)

        return torch.cat([rgb, sigma], dim=-1)

