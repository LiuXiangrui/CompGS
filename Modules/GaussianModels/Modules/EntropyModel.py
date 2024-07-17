from typing import Iterator

import torch
from compressai.models import CompressionModel
from compressai.ops import quantize_ste
from einops import rearrange
from torch import nn as nn
from torch.nn import functional as F, Parameter

from .BasicBlocks import BatchEntropyBottleneck, BatchGaussianConditional, ResidualMLP


class ReferenceFeatureEntropyModel(CompressionModel):
    def __init__(self, ref_feats_dim: int, ref_hyper_dim: int) -> None:
        """
        Entropy model for reference features.
        :param ref_feats_dim: dimension of reference features.
        :param ref_hyper_dim: dimension of hyper-priors for reference features.
        """
        super().__init__()

        self.h_a = nn.Sequential(
            nn.Linear(in_features=ref_feats_dim, out_features=ref_hyper_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=ref_hyper_dim, out_features=ref_hyper_dim)
        )

        self.h_s = ResidualMLP(in_dim=ref_hyper_dim, internal_dim=ref_feats_dim, out_dim=2 * ref_feats_dim, num_res_layer=2)

        self.entropy_bottleneck = BatchEntropyBottleneck(channels=ref_hyper_dim)
        self.gaussian_conditional = BatchGaussianConditional(scale_table=None)

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param y: features to be compressed, shape (N, C).
        :return y_hat: quantized features, shape (N, C).
        :return y_likelihoods: likelihoods of quantized features, shape (N, C).
        :return z_likelihoods: likelihoods of quantized hyper-priors, shape (N, D).
        """
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians().squeeze()  # shape (D, )
        z_hat = quantize_ste(z - z_offset) + z_offset  # straight-through estimator
        gaussian_params = self.h_s(z_hat)  # shape (N, 2 * C)

        means_hat, scales_hat = torch.chunk(gaussian_params, dim=-1, chunks=2)

        _, y_likelihoods = self.gaussian_conditional(y, scales=F.relu(scales_hat), means=means_hat)
        y_hat = quantize_ste(y - means_hat) + means_hat

        return y_hat, y_likelihoods, z_likelihoods

    @torch.no_grad()
    def compress(self, y: torch.Tensor) -> tuple[bytes, bytes]:
        """
        Compress features and hyper-priors into strings.
        :param y: features to be compressed, shape (N, C).
        :return y_strings: strings of features.
        :return z_strings: strings of hyper-priors.
        """
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, N=y.shape[0])
        gaussian_params = self.h_s(z_hat)  # shape (N, 2 * C)

        means_hat, scales_hat = torch.chunk(gaussian_params, dim=-1, chunks=2)

        indices = self.gaussian_conditional.build_indexes(F.relu(scales_hat, inplace=True))
        y_strings = self.gaussian_conditional.compress(y, indices=indices, means=means_hat)

        return y_strings, z_strings

    @torch.no_grad()
    def decompress(self, z_strings: bytes, y_strings: bytes, N: int) -> torch.Tensor:
        """
        Decompress features from strings.
        :param z_strings: strings of hyper-priors.
        :param y_strings: strings of feats.
        :param N: number of reference features.
        :return y_hat: decompress features, shape (N, C).
        """
        z_hat = self.entropy_bottleneck.decompress(z_strings, N=N)
        gaussian_params = self.h_s(z_hat)  # shape (N, 2 * C)

        means_hat, scales_hat = torch.chunk(gaussian_params, dim=-1, chunks=2)

        indices = self.gaussian_conditional.build_indexes(F.relu(scales_hat, inplace=True))
        y_hat = self.gaussian_conditional.decompress(y_strings, indices=indices, means=means_hat)
        y_hat = y_hat.squeeze(dim=0)

        return y_hat

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Parameters for training excluding auxiliary parameters of entropy bottleneck.
        :param recurse: whether to include parameters of submodules.
        """
        parameters = set(n for n, p in self.named_parameters(recurse=recurse) if not n.endswith('.quantiles') and p.requires_grad)
        params_dict = dict(self.named_parameters(recurse=recurse))
        params = (params_dict[n] for n in sorted(list(parameters)))

        return params


class ResidualFeatureEntropyModel(CompressionModel):
    def __init__(self, ref_feats_dim: int, res_feats_dim: int, res_hyper_dim: int, derive_factor: int) -> None:
        """
        Entropy model for residual features.
        :param ref_feats_dim: dimension of reference features which are used as contextual priors.
        :param res_feats_dim: dimension of residual features.
        :param res_hyper_dim: dimension of hyper-priors for residual features.
        :param derive_factor: number of coupled primitives derived from each anchor primitive.
        """
        super().__init__()

        self.derive_factor = derive_factor

        res_feats_dim = res_feats_dim * derive_factor
        res_hyper_dim = res_hyper_dim * derive_factor

        self.h_a = nn.Sequential(
            nn.Linear(in_features=res_feats_dim, out_features=res_hyper_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=res_hyper_dim, out_features=res_hyper_dim)
        )

        self.h_s = ResidualMLP(in_dim=ref_feats_dim + res_hyper_dim, internal_dim=ref_feats_dim, out_dim=2 * res_feats_dim, num_res_layer=2)

        self.entropy_bottleneck = BatchEntropyBottleneck(channels=res_hyper_dim)
        self.gaussian_conditional = BatchGaussianConditional(scale_table=None)

    def forward(self, y: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param y: residual features to be compressed, shape (N, K, G).
        :param ctx: reference features as contextual priors, shape (N, C).
        :return y_hat: quantized residual features, shape (N * K, G).
        :return y_likelihoods: likelihoods of quantized residual features, shape (N * K, G).
        :return z_likelihoods: likelihoods of quantized hyper-priors, shape (N * K, D).
        """
        z = self.h_a(rearrange(y, 'n k g -> n (k g)'))
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians().squeeze()
        z_hat = quantize_ste(z - z_offset) + z_offset  # straight-through estimator
        z_likelihoods = rearrange(z_likelihoods, 'n (k g) -> (n k) g', k=self.derive_factor)

        gaussian_params = rearrange(self.h_s(torch.cat([z_hat, ctx], dim=1)), 'n (k g) -> n k g', k=self.derive_factor)
        means_hat, scales_hat = torch.chunk(gaussian_params, dim=-1, chunks=2)  # shape (N, K, G)

        means_hat = rearrange(means_hat, 'n k g -> (n k) g')
        scales_hat = rearrange(scales_hat, 'n k g -> (n k) g')
        y = rearrange(y, 'n k g -> (n k) g')

        _, y_likelihoods = self.gaussian_conditional(y, scales=F.relu(scales_hat), means=means_hat)
        y_hat = quantize_ste(y - means_hat) + means_hat

        return y_hat, y_likelihoods, z_likelihoods

    @torch.no_grad()
    def compress(self, y: torch.Tensor, ctx: torch.Tensor) -> tuple[bytes, bytes]:
        """
        Compress residual features and hyper-priors into strings.
        :param y: residual features to be compressed, shape (N, K, G).
        :param ctx: reference features as contextual priors, shape (N, C).
        :return y_strings: strings of residual features.
        :return z_strings: strings of hyper-priors.
        """
        z = self.h_a(rearrange(y, 'n k g -> n (k g)'))
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, N=z.shape[0])

        gaussian_params = rearrange(self.h_s(torch.cat([z_hat, ctx], dim=1)), 'n (k g) -> n k g', k=self.derive_factor)
        means_hat, scales_hat = torch.chunk(gaussian_params, dim=-1, chunks=2)  # shape (N, K, G)

        means_hat = rearrange(means_hat, 'n k g -> (n k) g')
        scales_hat = rearrange(scales_hat, 'n k g -> (n k) g')
        y = rearrange(y, 'n k g -> (n k) g')

        indices = self.gaussian_conditional.build_indexes(F.relu(scales_hat, inplace=True))
        y_strings = self.gaussian_conditional.compress(y, indices=indices, means=means_hat)

        return y_strings, z_strings

    @torch.no_grad()
    def decompress(self, z_strings: bytes, y_strings: bytes, ctx: torch.Tensor) -> torch.Tensor:
        """
        Decompress residual features from strings.
        :param z_strings: strings of hyper-priors.
        :param y_strings: strings of residual features.
        :param ctx: reference features as contextual priors, shape (N, C).
        :return y_hat: decompressed residual features, shape (N, K, G).
        """
        z_hat = self.entropy_bottleneck.decompress(z_strings, N=ctx.shape[0])

        gaussian_params = rearrange(self.h_s(torch.cat([z_hat, ctx], dim=1)), 'n (k g) -> n k g', k=self.derive_factor)
        means_hat, scales_hat = torch.chunk(gaussian_params, dim=-1, chunks=2)  # shape (N, K, G)

        means_hat = rearrange(means_hat, 'n k g -> (n k) g')
        scales_hat = rearrange(scales_hat, 'n k g -> (n k) g')

        indices = self.gaussian_conditional.build_indexes(F.relu(scales_hat, inplace=True))
        y_hat = self.gaussian_conditional.decompress(y_strings, indices=indices, means=means_hat)
        y_hat = rearrange(y_hat, '(n k) g -> n k g', k=self.derive_factor)

        return y_hat

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Parameters for training excluding auxiliary parameters of entropy bottleneck.
        :param recurse: whether to include parameters of submodules.
        """
        parameters = set(n for n, p in self.named_parameters(recurse=recurse) if not n.endswith('.quantiles') and p.requires_grad)
        params_dict = dict(self.named_parameters(recurse=recurse))
        params = (params_dict[n] for n in sorted(list(parameters)))

        return params


class ScaleEntropyModel(CompressionModel):
    def __init__(self, ref_feats_dim: int, scale_dim: int = 6, quant_step: float = 0.01) -> None:
        """
        Entropy model for scales
        :param ref_feats_dim: dimension of reference features, which are used as contextual priors.
        :param scale_dim: dimension of scale features.
        :param quant_step: quantization step for scales.
        """
        super().__init__()
        self.quant_step = nn.Parameter(torch.ones(scale_dim, dtype=torch.float32) * quant_step, requires_grad=True)

        self.h_s = ResidualMLP(in_dim=ref_feats_dim, internal_dim=ref_feats_dim, out_dim=2 * scale_dim, num_res_layer=2)

        self.gaussian_conditional = BatchGaussianConditional(scale_table=None)

    def forward(self, y: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param y: scales to be compressed, shape (N, D).
        :param ctx: reference features as contextual priors, shape (N, C).
        :return y_hat: quantized scales, shape (N, D).
        :return y_likelihoods: likelihoods of quantized scales, shape (N, D).
        """
        params = self.h_s(ctx)
        means_hat, scales_hat = torch.chunk(params, dim=-1, chunks=2)  # shape (N, D)

        y = y / self.quant_step
        _, y_likelihoods = self.gaussian_conditional(y, scales=F.relu(scales_hat), means=means_hat)
        y_hat = quantize_ste(y - means_hat) + means_hat
        y_hat = y_hat * self.quant_step

        return y_hat, y_likelihoods

    @torch.no_grad()
    def compress(self, y: torch.Tensor, ctx: torch.Tensor) -> bytes:
        """
        Compress scales into strings.
        :param y: scales to be compressed, shape (N, D).
        :param ctx: reference features as contextual priors, shape (N, C).
        :return y_strings: strings of scales.
        """
        params = self.h_s(ctx)
        means_hat, scales_hat = torch.chunk(params, dim=-1, chunks=2)  # shape (N, D)
        indices = self.gaussian_conditional.build_indexes(F.relu(scales_hat, inplace=True))

        y = y / self.quant_step
        y_strings = self.gaussian_conditional.compress(y, indices=indices, means=means_hat)

        return y_strings

    @torch.no_grad()
    def decompress(self, strings: bytes, ctx: torch.Tensor) -> torch.Tensor:
        """
        Decompress scales from strings.
        :param strings: strings of scales.
        :param ctx: reference features as contextual priors, shape (N, C).
        :return y_hat: decompressed scales, shape (N, D).
        """
        params = self.h_s(ctx)
        means_hat, scales_hat = torch.chunk(params, dim=-1, chunks=2)  # shape (N, D)
        indices = self.gaussian_conditional.build_indexes(F.relu(scales_hat, inplace=True))

        y_hat = self.gaussian_conditional.decompress(strings, indices=indices, means=means_hat)
        y_hat = y_hat * self.quant_step

        return y_hat


class EntropyModel(CompressionModel):
    def __init__(self, ref_feats_dim: int, ref_hyper_dim: int, res_feats_dim: int, res_hyper_dim: int, derive_factor: int):
        """
        Entropy model for Gaussian network.
        :param ref_feats_dim: dimension of reference features.
        :param ref_hyper_dim: dimension of hyper-priors for reference features.
        :param res_feats_dim: dimension of residual features.
        :param res_hyper_dim: dimension of hyper-priors for residual features.
        :param derive_factor: number of coupled primitives derived from each anchor primitive.
        """
        super().__init__()

        self.ref_entropy_model = ReferenceFeatureEntropyModel(ref_feats_dim=ref_feats_dim, ref_hyper_dim=ref_hyper_dim)
        self.res_entropy_model = ResidualFeatureEntropyModel(res_feats_dim=res_feats_dim, res_hyper_dim=res_hyper_dim, 
                                                             ref_feats_dim=ref_feats_dim, derive_factor=derive_factor)
        self.scale_entropy_model = ScaleEntropyModel(ref_feats_dim=ref_feats_dim)

        self.learnable_module_names = {'ref_entropy_model', 'res_entropy_model', 'scale_entropy_model'}

    def forward(self, ref_feats: torch.Tensor, res_feats: torch.Tensor, scaling_factors_before_exp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Quantize features and estimate likelihoods.
        :param ref_feats: reference features, shape (N, C).
        :param res_feats: residual features, shape (N, K, G).
        :param scaling_factors_before_exp: scaling factors before exponentiation, shape (N, 6).
        :return ref_feats: quantized reference features, shape (N, C).
        :return res_feats: quantized residual features, shape (N, K, G).
        :return scaling_factors_before_exp: quantized scaling factors before exp activation, shape (N, 6).
        :return likelihoods: likelihoods of all quantized features.
        """
        # quantize reference features and estimate likelihoods
        ref_feats, ref_feats_likelihoods, ref_hyper_likelihoods = self.ref_entropy_model(ref_feats)
        # quantize residual features and estimate likelihoods
        res_feats, res_feats_likelihoods, res_hyper_likelihoods = self.res_entropy_model(res_feats, ctx=ref_feats)
        # quantize scales and estimate likelihoods
        scaling_factors_before_exp, scales_likelihoods = self.scale_entropy_model(scaling_factors_before_exp, ctx=ref_feats)

        likelihoods = {
            'ref_feats': ref_feats_likelihoods, 'ref_hyper': ref_hyper_likelihoods,
            'res_feats': res_feats_likelihoods, 'res_hyper': res_hyper_likelihoods,
            'scales': scales_likelihoods
        }

        return ref_feats, res_feats, scaling_factors_before_exp, likelihoods

    @torch.no_grad()
    def compress(self, ref_feats: torch.Tensor, res_feats: torch.Tensor, scaling_factors_before_exp: torch.Tensor) -> dict[str, bytes]:
        """
        Compress reference features, residual features and scaling factors into strings.
        :param ref_feats: reference features, shape (N, C).
        :param res_feats: residual features, shape (N, K, G).
        :param scaling_factors_before_exp: scaling factors before exponentiation, shape (N, 6).
        :return strings: strings of all features.
        """
        ref_feats_strings, ref_hyper_strings = self.ref_entropy_model.compress(ref_feats)
        ref_feats = self.ref_entropy_model.decompress(z_strings=ref_hyper_strings, y_strings=ref_feats_strings, N=ref_feats.shape[0])
        res_feats_strings, res_hyper_strings = self.res_entropy_model.compress(res_feats, ctx=ref_feats)
        scale_strings = self.scale_entropy_model.compress(scaling_factors_before_exp, ctx=ref_feats)

        strings = {
            'ref_feats_strings': ref_feats_strings, 'ref_hyper_strings': ref_hyper_strings,
            'res_feats_strings': res_feats_strings, 'res_hyper_strings': res_hyper_strings,
            'scale_strings': scale_strings,
        }

        return strings

    @torch.no_grad()
    def decompress(self, ref_feats_strings: bytes, ref_hyper_strings: bytes, res_feats_strings: bytes, res_hyper_strings: bytes,
                   scale_strings: bytes, num_anchor_primitives: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompress reference features, residual features and scaling factors from strings.
        :param ref_feats_strings: strings of reference features.
        :param ref_hyper_strings: strings of reference hyper-priors.
        :param res_feats_strings: strings of residual features.
        :param res_hyper_strings: strings of residual hyper-priors.
        :param scale_strings: strings of scales.
        :param num_anchor_primitives: number of anchor primitives.
        :return ref_feats: decompressed reference features, shape (N, C).
        :return res_feats: decompressed residual features, shape (N, K, G).
        :return scaling_factors: decompressed scaling factors, shape (N, 6).
        """
        ref_feats = self.ref_entropy_model.decompress(z_strings=ref_hyper_strings, y_strings=ref_feats_strings, N=num_anchor_primitives)
        res_feats = self.res_entropy_model.decompress(z_strings=res_hyper_strings, y_strings=res_feats_strings, ctx=ref_feats)
        scaling_factors = self.scale_entropy_model.decompress(strings=scale_strings, ctx=ref_feats)

        return ref_feats, res_feats, scaling_factors

    @torch.no_grad()
    def quantize_scales(self, scaling_factors_before_exp: torch.Tensor, ref_feats: torch.Tensor) -> torch.Tensor:
        """
        Quantize scaling factors, used in pre-filtering invisible anchor primitives.
        :param scaling_factors_before_exp: scaling factors of anchor primitives before exp activation, shape (N, 6).
        :param ref_feats: reference features, shape (N, C).
        :return quantized_scales: quantized scales, shape (N, 6).
        """
        ref_feats, _, _ = self.ref_entropy_model(ref_feats)
        scaling_factors_before_exp, _ = self.scale_entropy_model(scaling_factors_before_exp, ctx=ref_feats)

        return scaling_factors_before_exp

    def aux_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Auxiliary parameters of entropy bottleneck for training.
        :param recurse: whether to include parameters of submodules.
        """
        parameters_aux = set(n for n, p in self.named_parameters(recurse=recurse) if n.endswith('.quantiles') and p.requires_grad)
        params_dict = dict(self.named_parameters(recurse=recurse))
        params_aux = (params_dict[n] for n in sorted(list(parameters_aux)))

        return params_aux