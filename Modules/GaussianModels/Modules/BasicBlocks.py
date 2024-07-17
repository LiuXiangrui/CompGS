import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


class BatchEntropyBottleneck(EntropyBottleneck):
    def __init__(self, channels: int):
        """
        Entropy bottleneck supporting parallel coding along batches.
        """
        super().__init__(channels=channels)

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress input tensor into bytes.
        :param x: input tensor, shape (N, C).
        """
        assert len(x.shape) == 2, f'Shape of input tensor should be NxC, but got {x.shape}.'

        indices = self._build_indexes(x.size())
        medians = self._get_medians().detach()
        medians = self._extend_ndims(medians, n=0)
        medians = medians.expand(x.size(0), -1)

        symbols = self.quantize(x, mode='symbols', means=medians)

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        strings = self.entropy_coder.encode_with_indexes(
            symbols.reshape(-1).int().tolist(), indices.reshape(-1).int().tolist(), self._quantized_cdf.tolist(),
            self._cdf_length.reshape(-1).int().tolist(), self._offset.reshape(-1).int().tolist()
        )

        return strings

    @torch.no_grad()
    def decompress(self, strings: bytes, N: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Decompress strings.
        :param strings: bytes to be decompressed.
        :param N: number of samples.
        :param dtype: data type of the output tensor.
        """
        output_size = (N, self._quantized_cdf.size(0))
        indices = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), n=0)
        medians = medians.expand(N, -1)

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        x = self.entropy_coder.decode_with_indexes(
            strings, indices.reshape(-1).int().tolist(), self._quantized_cdf.tolist(),
            self._cdf_length.reshape(-1).int().tolist(), self._offset.reshape(-1).int().tolist())
        x = torch.tensor(x, dtype=dtype, device=indices.device).reshape(N, -1)
        x = self.dequantize(x, means=medians)

        return x


class BatchGaussianConditional(GaussianConditional):
    def __init__(self, scale_table: list | tuple = None):
        """
        Gaussian conditional distribution supporting parallel coding along batches.
        """
        super().__init__(scale_table=scale_table)

    @torch.no_grad()
    def compress(self, x: torch.Tensor, indices: torch.Tensor, means: torch.Tensor = None) -> bytes:
        """
        Compress input 2D tensor into bytes.
        :param x: input tensor, shape (N, C).
        :param indices: indices tensor, shape (N, C).
        :param means: means tensor, shape (N, C).
        """
        assert len(x.shape) == 2, f'Shape of input tensor should be NxC, but got {x.shape}.'

        symbols = self.quantize(x, mode='symbols', means=means)

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        strings = self.entropy_coder.encode_with_indexes(
            symbols.reshape(-1).int().tolist(), indices.reshape(-1).int().tolist(), self._quantized_cdf.tolist(),
            self._cdf_length.reshape(-1).int().tolist(), self._offset.reshape(-1).int().tolist())

        return strings

    @torch.no_grad()
    def decompress(self, strings: bytes, indices: torch.Tensor, dtype: torch.dtype = torch.float32, means: torch.Tensor = None) -> torch.Tensor:
        """
        Decompress strings.
        :param strings: bytes to be decompressed.
        :param indices: indices tensor, shape (N, C).
        :param dtype: data type of the output tensor.
        :param means: means tensor, shape (N, C).
        :return:
        """
        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        N = indices.shape[0]

        x = self.entropy_coder.decode_with_indexes(
            strings, indices.reshape(-1).int().tolist(), self._quantized_cdf.tolist(),
            self._cdf_length.reshape(-1).int().tolist(), self._offset.reshape(-1).int().tolist())
        x = torch.tensor(x, dtype=dtype, device=indices.device).reshape(N, -1)
        x = self.dequantize(x, means=means)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        """
        Residual block.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=dim, out_features=dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, internal_dim: int, out_dim: int, num_res_layer: int = 1, tail_activation: nn.Module = None) -> None:
        """
        Residual multi-layer perceptron.
        :param in_dim: input dimension.
        :param internal_dim: internal dimension.
        :param out_dim: output dimension.
        :param num_res_layer: number of residual layers.
        :param tail_activation: activation function.
        """
        super().__init__()

        layers = [
            nn.Linear(in_features=in_dim, out_features=internal_dim),
            nn.LeakyReLU(inplace=True)
        ]
        for i in range(num_res_layer):
            layers.append(ResidualBlock(dim=internal_dim))
        layers.append(nn.Linear(in_features=internal_dim, out_features=out_dim))
        self.net = nn.Sequential(*layers)
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_features=in_dim, out_features=out_dim)
        self.tail_activation = tail_activation if tail_activation is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail_activation(self.net(x) + self.skip(x))