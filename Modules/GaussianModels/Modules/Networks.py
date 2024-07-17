import math
from typing import Iterator

import torch
from compressai.models import CompressionModel
from einops import rearrange
from torch.nn import Parameter

from .EntropyModel import EntropyModel
from .Prediction import PredictionNetwork
from ...Common import GaussianParameters, LearningRateUpdateConfig


class Networks(CompressionModel):
    def __init__(self, ref_feats_dim: int, ref_hyper_dim: int, res_feats_dim: int, res_hyper_dim: int, derive_factor: int) -> None:
        """
        Network to predict Gaussian primitives and estimate rate.
        :param ref_feats_dim: dimension of reference features.
        :param ref_hyper_dim: dimension of hyper-priors for reference features.
        :param res_feats_dim: dimension of residual features.
        :param res_hyper_dim: dimension of hyper-priors for residual features.
        :param derive_factor: number of coupled primitives derived from each anchor primitive.
        """
        super().__init__()

        self.prediction_net = PredictionNetwork(ref_feats_dim=ref_feats_dim, res_feats_dim=res_feats_dim, derive_factor=derive_factor)

        # init entropy network
        self.entropy_model = EntropyModel(ref_feats_dim=ref_feats_dim, ref_hyper_dim=ref_hyper_dim,
                                          res_feats_dim=res_feats_dim, res_hyper_dim=res_hyper_dim, derive_factor=derive_factor)

        self.learnable_module_names = {
            'prediction_net': self.prediction_net.learnable_module_names,
            'entropy_model': self.entropy_model.learnable_module_names
        }

    def forward(self, means: torch.Tensor, scaling_factors_before_exp: torch.Tensor,
                ref_feats: torch.Tensor, res_feats: torch.Tensor, cam_center: torch.Tensor, skip_quant: bool = False,
                ) -> tuple[GaussianParameters, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Predict Gaussian primitives and estimate rate.
        :param means: means of anchor primitives, shape (N, 3).
        :param scaling_factors_before_exp: scaling factors of anchor primitives before exp activation, shape (N, 6).
        :param ref_feats: reference features, shape (N, C).
        :param res_feats: residual features, shape (N, K, G).
        :param cam_center: camera center, shape (3, ).
        :param skip_quant: whether to skip quantization and likelihoods estimation.
        :return gaussian_primitives: predicted Gaussian primitives.
        :return pred_opacities: predicted opacities, shape (N * K, 1).
        :return coupled_primitive_mask: mask of coupled primitives, shape (N * K, 1).
        :return bpp: bits per pixel for each parameter to be compressed.
        """
        # quantize and calculate likelihoods
        if skip_quant:
            res_feats = rearrange(res_feats, 'n k g -> (n k) g')
            likelihoods = None
        else:
            ref_feats, res_feats, scaling_factors_before_exp, likelihoods = self.entropy_model(
                ref_feats=ref_feats, res_feats=res_feats, scaling_factors_before_exp=scaling_factors_before_exp)

        # generate Gaussian primitives
        gaussian_primitives, pred_opacities, coupled_primitive_mask = self.prediction_net(
            means=means, scaling_factors=torch.exp(scaling_factors_before_exp), ref_feats=ref_feats, res_feats=res_feats, cam_center=cam_center)

        if skip_quant:
            return gaussian_primitives, pred_opacities, coupled_primitive_mask, {}

        # calculate bits per element
        bpp = {key: torch.log(value).sum() / (-math.log(2) * value.numel()) for key, value in likelihoods.items()}

        return gaussian_primitives, pred_opacities, coupled_primitive_mask, bpp

    @torch.no_grad()
    def compress(self, ref_feats: torch.Tensor, res_feats: torch.Tensor, scaling_factors_before_exp: torch.Tensor) -> dict[str, bytes]:
        """
        Compress reference features, residual features and scaling factors.
        :param ref_feats: reference features, shape (N, C).
        :param res_feats: residual features, shape (N, K, G).
        :param scaling_factors_before_exp: scaling factors of anchor primitives before exp activation, shape (N, 6).
        :return compressed_strings: compressed strings.
        """
        return self.entropy_model.compress(ref_feats=ref_feats, res_feats=res_feats, scaling_factors_before_exp=scaling_factors_before_exp)

    @torch.no_grad()
    def decompress(self, ref_feats_strings: bytes, ref_hyper_strings: bytes, res_feats_strings: bytes, res_hyper_strings: bytes,
                   scale_strings: bytes, num_anchor_primitives: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompress reference features, residual features and scaling factors.
        :param ref_feats_strings: strings of reference features.
        :param ref_hyper_strings: strings of reference hyper-priors.
        :param res_feats_strings: strings of residual features.
        :param res_hyper_strings: strings of residual hyper-priors.
        :param scale_strings: strings of scales.
        :param num_anchor_primitives: number of anchor primitives.
        :return ref_feats: reference features, shape (N, C).
        :return res_feats: residual features, shape (N, K, G).
        :return scaling_factors_before_exp: scaling factors of anchor primitives before exp activation, shape (N, 6).
        """
        return self.entropy_model.decompress(
            ref_feats_strings=ref_feats_strings, ref_hyper_strings=ref_hyper_strings,
            res_feats_strings=res_feats_strings, res_hyper_strings=res_hyper_strings,
            scale_strings=scale_strings, num_anchor_primitives=num_anchor_primitives)

    @torch.no_grad()
    def pred_gaussian_means(self, means: torch.Tensor, scaling_factors_before_exp: torch.Tensor,
                            ref_feats: torch.Tensor, res_feats: torch.Tensor) -> torch.Tensor:
        """
        Predict means of Gaussian primitives, used in adaptive control.
        :param means: means of anchor primitives, shape (N, 3).
        :param scaling_factors_before_exp: scaling factors of anchor primitives before exp activation, shape (N, 6).
        :param ref_feats: reference features, shape (N, C).
        :param res_feats: residual features, shape (N, K, G).
        :return means: means of predicted Gaussian primitives, shape (N * K, 3).
        """
        # quantize
        ref_feats, res_feats, scaling_factors_before_exp, _ = self.entropy_model(
            ref_feats=ref_feats, res_feats=res_feats, scaling_factors_before_exp=scaling_factors_before_exp)

        # predict means of Gaussian primitives
        return self.prediction_net.pred_gaussian_means(
            means=means, scaling_factors_before_exp=scaling_factors_before_exp, ref_feats=ref_feats, res_feats=res_feats)

    @torch.no_grad()
    def quantize_scales(self, scaling_factors_before_exp: torch.Tensor, ref_feats: torch.Tensor) -> torch.Tensor:
        """
        Quantize scaling factors of anchor primitives, used in pre-filtering invisible anchor Gaussians.
        :param scaling_factors_before_exp: scaling factors of anchor primitives before exp activation, shape (N, 6).
        :param ref_feats: reference features, shape (N, C).
        :return quantized_scales: quantized scales, shape (N, 6).
        """
        return self.entropy_model.quantize_scales(scaling_factors_before_exp=scaling_factors_before_exp, ref_feats=ref_feats)

    def aux_parameters(self) -> Iterator[Parameter]:
        """
        Auxiliary parameters for training.
        """
        return self.entropy_model.aux_parameters()

    def get_lr_param_pairs(self, lr_config_groups: dict) -> list[dict]:
        """
        Get learning rate parameters for optimizer initialization.
        """
        lr_param_pairs = [
            {
                'name': f'{param_cls}_{name}',
                'lr': lr_config_groups[param_cls][name + '_lr_init'],
                'params': getattr(getattr(self, param_cls), name).parameters()
            }
            for param_cls in self.learnable_module_names
            for name in self.learnable_module_names[param_cls]
        ]

        return lr_param_pairs

    def get_lr_aux_param_pairs(self, lr: float = 1e-4) -> list[dict]:
        """
        Get learning rate parameters for auxiliary optimizer initialization.
        """
        return [{'lr': lr, 'params': self.aux_parameters()}]

    def get_lr_update_configs(self, lr_config_groups: dict) -> dict[str, LearningRateUpdateConfig]:
        """
        Get learning rate update configuration.
        """
        lr_update_configs = {
            f'{param_cls}_{name}': LearningRateUpdateConfig(
                lr_init=lr_config_groups[param_cls][name + '_lr_init'],
                lr_delay_mult=lr_config_groups[param_cls][name + '_lr_delay_mult'],
                lr_final=lr_config_groups[param_cls][name + '_lr_final'],
            )
            for param_cls in self.learnable_module_names
            for name in self.learnable_module_names[param_cls]
        }
        return lr_update_configs