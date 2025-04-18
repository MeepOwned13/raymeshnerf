import torch
from torch import Tensor, nn
from .mlhhe import MultiLevelHybridHashEncoding
from .ogfilter import OccupancyGridFilter


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
    """NeRF as per the original paper"""

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
        with torch.no_grad():
            # Proposed fix for starting in local minima
            self.sigma_fc.bias.fill_(0.1)
            
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
        """Linear layers for RGB calculation"""

    def forward(self, coordinates: Tensor, directions: Tensor, skip_colors: bool = False) -> Tensor:
        """Perform RGBS calculation

        coordinates and directions must have the same dimensions for *...

        Args:
            coordinates (shape[*..., in_coordinates]): Input point coordinates
            directions (shape[*..., in_directions]): Input directions
            skip_colors: Skip color calculation?

        Returns:
            Tensor: RGBS (skip_colors=True) or Sigma (skip_colors=False)
                - **rgbs**: *shape[\*..., 4]*: RGB&Sigma
                - **sigma**: *shape[\*...]*: Sigma
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


class SphericalHarmonicsBasisEncoding(nn.Module):
    """Cartesian Coordinates to Spherical Harmonics Basis Encoding Module"""

    def __init__(self, degree: int = 3):
        """Init

        Args:
            degree: Degree of basis to encode to (0-3)
        """
        super(SphericalHarmonicsBasisEncoding, self).__init__()
        self.SIZES = (1, 4, 9, 16)
        """Possible last dim length of outputs ordered by degree"""

        assert degree in (0, 1, 2, 3)  # degree can be 0, 1, 2 or 3
        self.degree = degree
        """Degree of basis"""

    def forward(self, input: Tensor):
        """Perform Spherical Harmonics Basis Encoding

        Args:
            input (shape[..., 3]): Cartesian Coordinates to encode

        Returns:
            encoded (shape[..., SIZES[degree]]): Encoded tensor
        """
        encoded = torch.zeros((input.shape[:-1]) + (self.out_dim,), dtype=torch.float32, device=input.device)

        pi = torch.tensor([torch.pi], dtype=torch.float32, device=input.device)
        x, y, z = input[..., 0], input[..., 1], input[..., 2]
        r = torch.sqrt(x**2 + y**2 + z**2)

        if self.degree >= 0:
            encoded[..., 0] = 0.5 * torch.sqrt(1 / pi)

        if self.degree >= 1:
            encoded[..., 1] = torch.sqrt(3 / (4 * pi)) * y / r
            encoded[..., 2] = torch.sqrt(3 / (4 * pi)) * z / r
            encoded[..., 3] = torch.sqrt(3 / (4 * pi)) * x / r

        if self.degree >= 2:
            encoded[..., 4] = 1 / 2 * torch.sqrt(15 / pi) * (x * y) / r**2
            encoded[..., 5] = 1 / 2 * torch.sqrt(15 / pi) * (y * z) / r**2
            encoded[..., 6] = 1 / 4 * torch.sqrt(5 / pi) * (3 * z**2 - r**2) / r**2
            encoded[..., 7] = 1 / 2 * torch.sqrt(15 / pi) * (x * z) / r**2
            encoded[..., 8] = 1 / 2 * torch.sqrt(15 / pi) * (x**2 - y**2) / r**2

        if self.degree >= 3:
            encoded[..., 9] = 1 / 4 * torch.sqrt(35 / (2 * pi)) * y * (3 * x**2 - y**2) / r**3
            encoded[..., 10] = 1 / 2 * torch.sqrt(105 / pi) * (x * y * z) / r**3
            encoded[..., 11] = 1 / 4 * torch.sqrt(21 / (2 * pi)) * y * (5 * z**2 - r**2) / r**3
            encoded[..., 12] = 1 / 4 * torch.sqrt(7 / pi) * z * (5 * z**2 - 3 * r**2) / r**3
            encoded[..., 13] = 1 / 4 * torch.sqrt(21 / (2 * pi)) * x * (5 * z**2 - r**2) / r**3
            encoded[..., 14] = 1 / 4 * torch.sqrt(105 / pi) * z * (x**2 - y**2) / r**3
            encoded[..., 15] = 1 / 4 * torch.sqrt(35 / (2 * pi)) * x * (x**2 - 3 * y**2) / r**3

        return encoded

    @property
    def out_dim(self):
        """Length of last dimension of output"""
        return self.SIZES[self.degree]


class InstantNGP(nn.Module):
    """InstantNGP implementation using MLHHE from https://github.com/cheind"""

    def __init__(self, hidden_size: int = 64, encoding_log2: int = 19, embed_dims: int = 2, levels: int = 16,
                 min_res: int = 32, max_res: int = 512, max_res_dense: int = 256, f_res: int = 128,
                 f_sigma_init: float = 0.04, f_sigma_threshold: float = 0.01, f_stochastic_test: bool = True,
                 f_update_decay: float = 0.7, f_update_noise_scale: float = None,
                 f_update_selection_rate: float = 0.25):
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
        """
        super(InstantNGP, self).__init__()
        self.filter = OccupancyGridFilter(
            res=f_res,
            density_initial=f_sigma_init,
            density_threshold=f_sigma_threshold,
            stochastic_test=f_stochastic_test,
            update_decay=f_update_decay,
            update_noise_scale=f_update_noise_scale,
            update_selection_rate=f_update_selection_rate,
        )
        """Occupancy Grid Filtering for coordinates"""

        self.mlhhe = MultiLevelHybridHashEncoding(
            n_encodings=2 ** encoding_log2,
            n_input_dims=3,
            n_embed_dims=embed_dims,
            n_levels=levels,
            min_res=min_res,
            max_res=max_res,
            max_n_dense=max_res_dense ** 3,
        )
        """Multi-Level Hybrid Hash Encoding used for coordinate encoding"""

        self.feature_mlp = nn.Sequential(
            nn.Linear(levels * embed_dims, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 16),  # index 0 is log density
        )
        """MLP processing encoded coordinates, output at index 0 is log of sigma"""

        nn.init.constant_(self.feature_mlp[-1].bias[:1], -1.0)

        self.direction_encoder = SphericalHarmonicsBasisEncoding(3)
        """Encoder for direction vectors"""

        self.rgb_mlp = nn.Sequential(
            nn.Linear(16 + 16, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 3),
            nn.Sigmoid(),
        )
        """MLP processing features and directions to generate RGB values"""

    def forward(self, coordinates: Tensor, directions: Tensor, skip_colors: bool = False, masked: bool = True):
        """Perform RGBS calculation

        coordinates and directions must have the same dimensions for *...

        Args:
            coordinates (shape[*..., in_coordinates]): Input point coordinates
            directions (shape[*..., in_directions]): Input directions
            skip_colors: Skip color calculation?
            masked: Use occupancy grid filtering?

        Returns:
            Tensor: RGBS (skip_colors=True) or Sigma (skip_colors=False)
                - **rgbs**: *shape[\*..., 4]*: RGB&Sigma
                - **sigma**: *shape[\*...]*: Sigma
        """
        device = coordinates.device
        out_shape = list(coordinates.shape[:-1])

        coordinates = coordinates.reshape(-1, 3)
        sigma = torch.zeros([coordinates.shape[0], 1], dtype=torch.float32, device=device)
        rgb = torch.zeros([coordinates.shape[0], 3], dtype=torch.float32, device=device)

        mask = torch.ones(coordinates.shape[0], dtype=bool, device=device)
        if masked:
            mask = self.filter.test(coordinates)
        coordinates = coordinates[mask]

        embeds = self.mlhhe(coordinates).flatten(1, -1)
        features = self.feature_mlp(embeds)
        sigma[mask] = torch.exp(features[..., 0:1])

        if skip_colors:
            return sigma.squeeze(-1).reshape(out_shape)

        directions = directions.reshape(-1, 3)
        features = torch.cat([
            features,
            self.direction_encoder(directions[mask])
        ], dim=-1)

        rgb[mask] = self.rgb_mlp(features)
        rgbs = torch.cat([rgb, sigma], dim=-1).reshape(out_shape + [-1])
        return rgbs

    def update_filter(self):
        self.filter.update(self)
