import os
import time

import numpy as np
import torch
import torch.nn as nn
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from .Modules import Networks
from .Modules import Parameters

from ..Common import CustomLogger
from ..Common import RenderSettings, RenderResults
from ..Common import calculate_morton_order, compress_gpcc, decompress_gpcc
from ..Common import load_point_clouds, voxelize_sample, adaptive_voxel_size


class GaussianModel(nn.Module):
    def __init__(self, logger: CustomLogger, device: str, voxel_size: float, derive_factor: int,
                 ref_feats_dim: int, ref_hyper_dim: int, res_feats_dim: int, res_hyper_dim: int,
                 background_color: float = 0.) -> None:
        """
        Gaussian model.
        :param logger: logger.
        :param device: device of Gaussian model.
        :param voxel_size: size of voxels.
        :param derive_factor: number of coupled primitives derived from each anchor primitive.
        :param ref_feats_dim: dimension of reference features.
        :param ref_hyper_dim: dimension of hyper-priors corresponding to reference features.
        :param res_feats_dim: dimension of residual features.
        :param res_hyper_dim: dimension of hyper-priors corresponding to residual features.
        :param background_color: background color, default is black (0).
        """
        super().__init__()

        self.logger, self.device, self.voxel_size = logger, device, voxel_size
        self.background_colors = torch.ones(3, dtype=torch.float32, device=self.device, requires_grad=False) * background_color  # background color, default is black

        self.network = Networks(
            ref_feats_dim=ref_feats_dim, ref_hyper_dim=ref_hyper_dim,
            res_feats_dim=res_feats_dim, res_hyper_dim=res_hyper_dim,
            derive_factor=derive_factor).to(self.device)

        self.gaussian_params = Parameters(derive_factor=derive_factor, ref_feats_dim=ref_feats_dim, res_feats_dim=res_feats_dim, device=device)

        self.init_params = False  # whether Gaussian model parameters are initialized

    @property
    def num_anchor_primitive(self) -> int:
        """
        Return number of anchor primitives.
        """
        return self.gaussian_params.num_anchor_primitive

    @property
    def num_coupled_primitive(self) -> int:
        """
        Return number of coupled primitives.
        """
        return self.gaussian_params.num_coupled_primitive

    @property
    def derive_factor(self) -> int:
        """
        Return number of coupled primitives derived from each anchor primitive.
        """
        return self.gaussian_params.derive_factor

    @property
    def res_feats_dim(self) -> int:
        """
        Return dimension of residual features.
        """
        return self.gaussian_params.res_feats_dim

    @property
    def means(self) -> torch.Tensor:
        """
        Return means of anchor primitives with shape (N, 3).
        """
        return self.gaussian_params.means

    @property
    def scaling_factors_before_exp(self) -> torch.Tensor:
        """
        Return scaling factors before exponentiation with shape (N, 6).
        """
        return self.gaussian_params.scales_before_exp

    @property
    def rotations(self) -> torch.Tensor:
        """
        Return normalized rotation quaternion with shape (N, 4).
        """
        return self.gaussian_params.rotations

    @property
    def ref_feats(self) -> torch.Tensor:
        """
        Return reference features with shape (N, C).
        """
        return self.gaussian_params.ref_feats

    @property
    def res_feats(self) -> torch.Tensor:
        """
        Return residual features with shape (N, K, D).
        """
        return self.gaussian_params.res_feats

    @property
    def accumulated_grads(self) -> torch.Tensor:
        """
        Return accumulated gradients of predicted Gaussian primitives with shape (N, 1).
        """
        return self.gaussian_params.accumulated_grads

    @property
    def coupled_denorm(self) -> torch.Tensor:
        """
        Return times of coupled primitives accessed in rendering with shape (N * K, 1).
        """
        return self.gaussian_params.coupled_denorm

    @property
    def accumulated_opacities(self) -> torch.Tensor:
        """
        Return accumulated opacities of predicted Gaussian primitives with shape (N, 1).
        """
        return self.gaussian_params.accumulated_opacities

    @property
    def anchor_denorm(self) -> torch.Tensor:
        """
        Return times of anchor primitives accessed in rendering with shape (N, 1).
        """
        return self.gaussian_params.anchor_denorm

    @property
    @torch.no_grad()
    def pred_gaussian_means(self) -> torch.Tensor:
        """
        Return predicted means of Gaussian primitives with shape (N * K, 3), used in adaptive control.
        """
        return self.network.pred_gaussian_means(
            means=self.means, scaling_factors_before_exp=self.scaling_factors_before_exp,
            ref_feats=self.ref_feats, res_feats=self.res_feats)

    @property
    @torch.no_grad()
    def anchor_primitive_scales(self) -> torch.Tensor:
        """
        Return scales of anchor primitives with shape (N, 3), used in pre-filtering.
        """
        quant_scaling_factor = self.network.quantize_scales(scaling_factors_before_exp=self.scaling_factors_before_exp, ref_feats=self.ref_feats)
        return torch.exp(quant_scaling_factor[:, :3])

    @property
    def aux_loss(self) -> torch.Tensor:
        """
        Return auxiliary loss of entropy bottlenecks.
        """
        return self.network.aux_loss()

    def forward(self):
        raise NotImplementedError('Please do not call the forward method directly, instead call the render method.')

    def render(self, render_settings: RenderSettings, scaling_modifier: float = 1., retain_grad: bool = False, skip_quant: bool = False) -> RenderResults:
        """
        Render, used in training.
        :param render_settings: render settings.
        :param scaling_modifier: factor multiplied to scaling vectors.
        :param retain_grad: whether to retain gradients.
        :param skip_quant: whether to skip quantization.
        """
        assert self.init_params, 'Gaussian model parameters are not initialized'

        # create rasterizer
        raster_settings = GaussianRasterizationSettings(
            image_height=render_settings.image_height, image_width=render_settings.image_width,
            tanfovx=render_settings.tanfovx, tanfovy=render_settings.tanfovy, bg=self.background_colors,
            scale_modifier=scaling_modifier, viewmatrix=render_settings.viewmatrix, projmatrix=render_settings.projmatrix,
            sh_degree=1, campos=render_settings.campos, prefiltered=False, debug=False)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # prefilter invisible anchor primitives
        radii_pure = rasterizer.visible_filter(means3D=self.means, scales=self.anchor_primitive_scales, rotations=self.rotations, cov3D_precomp=None)
        anchor_primitive_visible_mask = radii_pure > 0.
        assert anchor_primitive_visible_mask.shape[0] == self.num_anchor_primitive, f'Number of visible masks {anchor_primitive_visible_mask.shape[0]} should be the same as number of anchor primitives {self.num_anchor_primitive}.'

        means, scaling_factors_before_exp = self.means[anchor_primitive_visible_mask], self.scaling_factors_before_exp[anchor_primitive_visible_mask]
        ref_feats, res_feats = self.ref_feats[anchor_primitive_visible_mask], self.res_feats[anchor_primitive_visible_mask]

        # predict Gaussian primitives
        gaussian_primitives, pred_opacities, coupled_primitive_mask, bpp = self.network(
            means=means, scaling_factors_before_exp=scaling_factors_before_exp, ref_feats=ref_feats, res_feats=res_feats,
            cam_center=render_settings.campos, skip_quant=skip_quant)

        # render
        projected_means = torch.zeros_like(gaussian_primitives.means, requires_grad=True) + 0.  # means of Gaussian primitives projected into camera space with shape (N, 3)
        if retain_grad:
            projected_means.retain_grad()

        rendered_img, radii = rasterizer(
            means3D=gaussian_primitives.means, means2D=projected_means,
            scales=gaussian_primitives.scales, rotations=gaussian_primitives.rotations, cov3D_precomp=None,
            shs=None, colors_precomp=gaussian_primitives.colors, opacities=gaussian_primitives.opacities)

        render_results = RenderResults(
            rendered_img=torch.clamp(rendered_img, min=0., max=1.),  # clamp to [0, 1]
            projected_means=projected_means,
            visibility_mask=radii > 0,
            pred_opacities=pred_opacities,
            scales=gaussian_primitives.scales,
            anchor_primitive_visible_mask=anchor_primitive_visible_mask,
            coupled_primitive_mask=coupled_primitive_mask,
            bpp=bpp
        )

        return render_results

    @torch.no_grad()
    def render_inference(self, render_settings: RenderSettings, scaling_modifier: float = 1.) -> tuple[torch.Tensor, float, float]:
        """
        Render, used in inference.
        :param render_settings: render settings.
        :param scaling_modifier: factor multiplied to scaling vectors.
        :return rendered_img: rendered image.
        :return predict_time: time of predicting Gaussian primitives in seconds.
        :return render_time: time of rendering Gaussian primitives in milliseconds.
        """
        assert self.init_params, 'Gaussian model parameters are not initialized'
        assert not self.training, 'Gaussian model is in training mode.'

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        # create rasterizer
        raster_settings = GaussianRasterizationSettings(
            image_height=render_settings.image_height, image_width=render_settings.image_width,
            tanfovx=render_settings.tanfovx, tanfovy=render_settings.tanfovy, bg=self.background_colors,
            scale_modifier=scaling_modifier, viewmatrix=render_settings.viewmatrix,
            projmatrix=render_settings.projmatrix,
            sh_degree=1, campos=render_settings.campos, prefiltered=False, debug=False)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # prefilter invisible anchor primitives
        radii_pure = rasterizer.visible_filter(means3D=self.means, scales=torch.exp(self.scaling_factors_before_exp[:, :3]), rotations=self.rotations, cov3D_precomp=None)
        anchor_primitive_visible_mask = radii_pure > 0.

        means, scaling_factors_before_exp = self.means[anchor_primitive_visible_mask], self.scaling_factors_before_exp[anchor_primitive_visible_mask]
        ref_feats, res_feats = self.ref_feats[anchor_primitive_visible_mask], self.res_feats[anchor_primitive_visible_mask]

        # predict Gaussian primitives
        gaussian_primitives, _, _, _ = self.network(
            means=means, scaling_factors_before_exp=scaling_factors_before_exp, ref_feats=ref_feats, res_feats=res_feats,
            cam_center=render_settings.campos, skip_quant=True)

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        predict_time = (end_time - start_time)  # seconds

        # render
        projected_means = torch.zeros_like(gaussian_primitives.means, requires_grad=True) + 0.  # means of Gaussian primitives projected into camera space with shape (N, 3)

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        rendered_img, _ = rasterizer(
            means3D=gaussian_primitives.means, means2D=projected_means,
            scales=gaussian_primitives.scales, rotations=gaussian_primitives.rotations, cov3D_precomp=None,
            shs=None, colors_precomp=gaussian_primitives.colors, opacities=gaussian_primitives.opacities)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        render_time = (end_time - start_time) * 1000  # milliseconds

        rendered_img = torch.clamp(rendered_img, min=0., max=1.)  # clamp to [0, 1]

        return rendered_img, predict_time, render_time

    def init_from_sfm(self, sfm_point_cloud_path: str) -> None:
        """
        Initialize primitives from sparse point clouds.
        :param sfm_point_cloud_path: path of sparse point clouds.
        """
        # load point clouds
        sfm_points, sfm_colors = load_point_clouds(point_cloud_path=sfm_point_cloud_path)
        assert sfm_points.shape[0] == sfm_colors.shape[0], 'Number of points and colors should be the same.'
        assert sfm_points.shape[1] == sfm_colors.shape[1] == 3, 'Only support 3D points and RGB colors.'
        self.logger.info(f'Load sparse point clouds from {sfm_point_cloud_path}...')

        self.voxel_size = self.voxel_size or adaptive_voxel_size(sfm_points)

        # init Gaussian data
        points = voxelize_sample(sfm_points, voxel_size=self.voxel_size)
        self.gaussian_params.init_from_sparse_points(points=points)

        self.init_params = True
        self.logger.info(f'Initialized {self.num_anchor_primitive} anchor primitives by sparse points and voxel size {self.voxel_size}.')

    @torch.no_grad()
    def save_compressed_params(self, npz_path: str, gpcc_codec_path: str) -> dict[str, float]:
        """
        Save Gaussian model parameters to binary file.
        :param npz_path: path of binary file.
        :param gpcc_codec_path: path of GPCC codec used to compress means of anchor primitives.
        """
        assert self.init_params, 'Gaussian model parameters are not initialized.'
        assert npz_path.endswith('.npz'), 'Only support npz file.'

        means, scaling_factors_before_exp = self.means, self.scaling_factors_before_exp
        ref_feats, res_feats = self.ref_feats, self.res_feats

        # voxelize means
        means = torch.round(means / self.voxel_size)
        assert torch.unique(means, dim=0).shape[0] == means.shape[0], 'Duplicated means detected.'  # there should be no duplicated means

        # sorted all parameters by the Morton order of means
        sorted_indices = calculate_morton_order(means)
        means, scaling_factors_before_exp = means[sorted_indices], scaling_factors_before_exp[sorted_indices]
        ref_feats, res_feats = ref_feats[sorted_indices], res_feats[sorted_indices]

        # compress features
        strings = self.network.compress(ref_feats=ref_feats, res_feats=res_feats, scaling_factors_before_exp=scaling_factors_before_exp)

        # compress means by G-PCC
        means_strings = compress_gpcc(means, gpcc_codec_path=gpcc_codec_path)

        strings['means_strings'] = means_strings

        # save to bin file
        np.savez_compressed(npz_path, voxel_size=self.voxel_size, **strings)

        # collect bitstream size in MB
        size = {f'{key}_size': len(value) / 1024 / 1024 for key, value in strings.items()}
        size['total'] = os.path.getsize(npz_path) / 1024 / 1024

        self.logger.info(f'\nSave compressed Gaussian model parameters to {npz_path}...\n')

        return size

    @torch.no_grad()
    def load_compressed_params(self, npz_path: str, gpcc_codec_path: str) -> None:
        """
        Load Gaussian model parameters from binary file.
        :param npz_path: path of binary file.
        :param gpcc_codec_path: path of GPCC codec used to decompress means of anchor primitives.
        """
        assert npz_path.endswith('.npz'), 'Only support npz file.'

        # load from bin file
        data_dict = np.load(npz_path)

        self.voxel_size = float(data_dict['voxel_size'])

        means_strings = data_dict['means_strings'].tobytes()
        scale_strings = data_dict['scale_strings'].tobytes()
        ref_feats_strings = data_dict['ref_feats_strings'].tobytes()
        ref_hyper_strings = data_dict['ref_hyper_strings'].tobytes()
        res_feats_strings = data_dict['res_feats_strings'].tobytes()
        res_hyper_strings = data_dict['res_hyper_strings'].tobytes()

        # decompress means by G-PCC
        means = decompress_gpcc(means_strings, gpcc_codec_path=gpcc_codec_path).to(self.device)
        sorted_indices = calculate_morton_order(means)

        # decompress features
        ref_feats, res_feats, scaling_factors_before_exp = self.network.decompress(
            scale_strings=scale_strings, num_anchor_primitives=means.shape[0],
            ref_feats_strings=ref_feats_strings, ref_hyper_strings=ref_hyper_strings,
            res_feats_strings=res_feats_strings, res_hyper_strings=res_hyper_strings)

        # sorted means by the Morton order
        means = means[sorted_indices]

        # devoxelize means
        means = means * self.voxel_size

        # create rotation quaternions
        rotations = torch.zeros((means.shape[0], 4), dtype=torch.float32, device=self.device)
        rotations[:, 0] = 1.

        self.replace_params(param_name='means', param_value=nn.Parameter(means, requires_grad=True))
        self.replace_params(param_name='scales_before_exp', param_value=nn.Parameter(scaling_factors_before_exp, requires_grad=True))
        self.replace_params(param_name='rotations_before_norm', param_value=nn.Parameter(rotations, requires_grad=True))
        self.replace_params(param_name='ref_feats', param_value=nn.Parameter(ref_feats, requires_grad=True))
        self.replace_params(param_name='res_feats', param_value=nn.Parameter(res_feats, requires_grad=True))

        self.init_params = True
        self.logger.info(f'Load compressed Gaussian model parameters from {npz_path}...')

    @torch.no_grad()
    def save_uncompressed_params(self, ply_path: str) -> None:
        """
        Save Gaussian model parameters to ply file.
        """
        assert self.init_params, 'Gaussian model parameters are not initialized.'
        assert ply_path.endswith('.ply'), 'Only support ply file.'

        # save to ply file
        self.gaussian_params.save_uncompressed(ply_path=ply_path)
        self.logger.info(f'\nSave uncompressed Gaussian model parameters to {ply_path}...')

    @torch.no_grad()
    def load_uncompressed_params(self, ply_path: str) -> None:
        """
        Load Gaussian model parameters from ply file.
        """
        assert ply_path.endswith('.ply'), 'Only support ply file.'

        # load from ply file
        self.gaussian_params.load_uncompressed(ply_path=ply_path)

        self.init_params = True
        self.logger.info(f'\nLoad uncompressed Gaussian model parameters from {ply_path}...')

    def save_weights(self, weight_path: str, update_entropy_model: bool = False) -> None:
        """
        Save Gaussian model weights to file.
        :param weight_path: path of weight file.
        :param update_entropy_model: whether to update entropy model.
        """
        assert weight_path.endswith('.pth'), 'Only support pth file.'
        if update_entropy_model:
            assert self.network.update(), 'Entropy model is not updated.'

        state_dicts = self.network.state_dict()
        torch.save(state_dicts, weight_path)
        self.logger.info(f'Saving network weights to {weight_path}...')

    def load_weights(self, weight_path: str) -> None:
        """
        Load Gaussian model weights from file.
        """
        assert weight_path.endswith('.pth'), 'Only support pth file.'
        state_dicts = torch.load(weight_path, map_location=self.device)
        self.network.load_state_dict(state_dicts)
        self.logger.info(f'Loading network weights from {weight_path}...')

    def replace_params(self, param_name: str, param_value: nn.Parameter) -> None:
        """
        Replace Gaussian model parameters.
        """
        self.gaussian_params.replace_params(param_name=param_name, param_value=param_value)

    def update_aux_params(self, render_results: RenderResults) -> None:
        """
        Update auxiliary parameters based on rendering results.
        """
        self.gaussian_params.update_aux_params(render_results=render_results)

    def reset_aux_params(self) -> None:
        """
        Reset auxiliary parameters.
        """
        self.gaussian_params.reset_aux_params()

    def remove_aux_params(self) -> None:
        """
        Remove auxiliary parameters after adaptive control.
        """
        self.gaussian_params.remove_aux_params()

    def replace_aux_params(self, param_name: str, param_value: torch.Tensor) -> None:
        """
        Replace auxiliary parameters.
        """
        self.gaussian_params.replace_aux_params(param_name=param_name, param_value=param_value)