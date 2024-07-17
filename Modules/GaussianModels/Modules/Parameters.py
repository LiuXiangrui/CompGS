import numpy as np
import torch
from einops import repeat
from knn_dist import distance
from plyfile import PlyData, PlyElement
from torch import nn as nn
from torch.nn import functional as F

from ...Common import RenderResults, LearningRateUpdateConfig


class Parameters(nn.Module):
    def __init__(self, derive_factor: int, ref_feats_dim: int, res_feats_dim: int, device: str) -> None:
        """
        Gaussians data.
        :param derive_factor: number of coupled primitives derived from each anchor primitive.
        :param ref_feats_dim: dimension of reference features.
        :param res_feats_dim: dimension of residual features.
        :param device: device of data.
        """
        super().__init__()

        self.device = device
        self.derive_factor = derive_factor
        self.ref_feats_dim, self.res_feats_dim = ref_feats_dim, res_feats_dim

        # create model parameters
        self.means = torch.empty(0)  # means of anchor primitives, shape (N, 3).
        self.scales_before_exp = torch.empty(0)  # scaling vectors of anchor primitives before exp activation, shape (N, 6).
        self.rotations_before_norm = torch.empty(0)  # rotation quaternions of anchor primitives before normalization, shape (N, 4).
        self.ref_feats = torch.empty(0)  # reference features of anchor primitives, shape (N, ref_feats_dim).
        self.res_feats = torch.empty(0)  # residual features of coupled primitives, shape (N, K, res_feats_dim).

        # create auxiliary variables
        self.accumulated_opacities = torch.empty(0)  # accumulated opacities of predicted Gaussian primitives, shape (N, 1)
        self.anchor_denorm = torch.empty(0)  # times of anchor primitives accessed in rendering, shape (N, 1)
        self.accumulated_grads = torch.empty(0)  # accumulated gradients of predicted Gaussian primitives, shape (N * K, 1)
        self.coupled_denorm = torch.empty(0)  # times of coupled primitives accessed in rendering, shape (N * K, 1)

        # first name is the name of learning rate specification in the config file, the second name is the name of the parameter in the GaussianData class
        self.learnable_param_names = {
            'means': 'means', 'scales': 'scales_before_exp', 'rotations': 'rotations_before_norm',
            'ref_feats': 'ref_feats', 'res_feats': 'res_feats'
        }

    @property
    def num_anchor_primitive(self) -> int:
        """
        Return number of anchor primitives.
        """
        return self.means.shape[0]

    @property
    def num_coupled_primitive(self) -> int:
        """
        Return number of coupled primitives.
        """
        return self.means.shape[0] * self.derive_factor

    @property
    def rotations(self) -> torch.Tensor:
        """
        Return normalized rotation quaternion with shape (N, 4).
        """
        return F.normalize(self.rotations_before_norm)

    @torch.no_grad()
    def replace_params(self, param_name: str, param_value: nn.Parameter) -> None:
        """
        Replace Gaussian model parameters.
        """
        assert hasattr(self, param_name) and isinstance(param_value, nn.Parameter) and param_value.requires_grad, f'Illegal parameter {param_name}'
        assert getattr(self, param_name).numel() == 0 or getattr(self, param_name).shape[1:] == param_value.shape[1:], f'Dimension mismatch for {param_name}'
        setattr(self, param_name, param_value)

    @torch.no_grad()
    def update_aux_params(self, render_results: RenderResults) -> None:
        """
        Update auxiliary parameters based on rendering results.
        """
        grad_norm = torch.norm(render_results.projected_means.grad[render_results.visibility_mask, :2], dim=-1, keepdim=True)  # shape (N * K * mask, 1)

        pred_opacities = torch.clamp_min_(render_results.pred_opacities.clone().detach().reshape(-1, self.derive_factor), min=0.)

        anchor_primitive_visible_mask = render_results.anchor_primitive_visible_mask  # shape (N, 1)

        combined_mask = torch.zeros_like(self.accumulated_grads, dtype=torch.bool).squeeze(dim=1)
        # exclude coupled primitives not used in rendering
        combined_mask[repeat(anchor_primitive_visible_mask, 'n -> (n k)', k=self.derive_factor)] = render_results.coupled_primitive_mask
        combined_mask[combined_mask.clone()] = render_results.visibility_mask

        # update auxiliary variables
        self.anchor_denorm[anchor_primitive_visible_mask] += 1.
        self.accumulated_opacities[anchor_primitive_visible_mask] += pred_opacities.sum(dim=1, keepdim=True)

        self.coupled_denorm[combined_mask] += 1.
        self.accumulated_grads[combined_mask] += grad_norm

    @torch.no_grad()
    def reset_aux_params(self) -> None:
        """
        Reset auxiliary parameters.
        """
        self.accumulated_grads = torch.zeros(self.num_coupled_primitive, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.coupled_denorm = torch.zeros(self.num_coupled_primitive, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.accumulated_opacities = torch.zeros(self.num_anchor_primitive, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.anchor_denorm = torch.zeros(self.num_anchor_primitive, 1, dtype=torch.float32, device=self.device, requires_grad=False)

        torch.cuda.empty_cache()

    @torch.no_grad()
    def remove_aux_params(self) -> None:
        """
        Remove auxiliary parameters.
        """
        self.anchor_denorm = torch.empty(0)
        self.accumulated_opacities = torch.empty(0)
        self.coupled_denorm = torch.empty(0)
        self.accumulated_grads = torch.empty(0)

        torch.cuda.empty_cache()

    @torch.no_grad()
    def replace_aux_params(self, param_name: str, param_value: torch.Tensor) -> None:
        """
        Replace auxiliary parameters.
        """
        assert hasattr(self, param_name) and isinstance(param_value, torch.Tensor), f'Illegal parameter {param_name}.'
        assert param_value.requires_grad is False, f'Parameter {param_name} should not require grad.'
        setattr(self, param_name, param_value)

    @torch.no_grad()
    def init_from_sparse_points(self, points: np.ndarray):
        """
        Initialize Gaussian model parameters from sparse points.
        """
        points = torch.tensor(np.asarray(points), dtype=torch.float32, device=self.device)  # shape (N, 3)

        dists = torch.clamp_min(distance(points), min=1e-7)
        scales_before_exp = torch.log(repeat(torch.sqrt(dists), 'n -> n k', k=6))  # shape (N, 6)

        rotations_before_norm = torch.zeros((points.shape[0], 4), dtype=torch.float32, device=self.device)  # shape (N, 4)
        rotations_before_norm[:, 0] = 1.0

        ref_feats = torch.zeros(points.shape[0], self.ref_feats_dim, dtype=torch.float32, device=self.device)
        res_feats = torch.zeros(points.shape[0], self.derive_factor, self.res_feats_dim, dtype=torch.float32, device=self.device)

        # init parameters
        self.replace_params(param_name='means', param_value=nn.Parameter(points, requires_grad=True))
        self.replace_params(param_name='scales_before_exp', param_value=nn.Parameter(scales_before_exp, requires_grad=True))
        self.replace_params(param_name='rotations_before_norm', param_value=nn.Parameter(rotations_before_norm, requires_grad=True))
        self.replace_params(param_name='ref_feats', param_value=nn.Parameter(ref_feats, requires_grad=True))
        self.replace_params(param_name='res_feats', param_value=nn.Parameter(res_feats, requires_grad=True))

        # init auxiliary variables
        self.accumulated_grads = torch.zeros(self.num_coupled_primitive, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.coupled_denorm = torch.zeros(self.num_coupled_primitive, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.accumulated_opacities = torch.zeros(self.num_anchor_primitive, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.anchor_denorm = torch.zeros(self.num_anchor_primitive, 1, dtype=torch.float32, device=self.device, requires_grad=False)

    @torch.no_grad()
    def save_uncompressed(self, ply_path: str) -> None:
        """
        Save uncompressed Gaussian model parameters to ply file.
        """
        # convert parameters to numpy array
        attributes_name = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        attributes_name.extend(['f_res_feat_{}'.format(i) for i in range(self.res_feats.shape[1] * self.res_feats.shape[2])])
        attributes_name.extend(['f_ref_feat_{}'.format(i) for i in range(self.ref_feats.shape[1])])
        attributes_name.extend(['scale_{}'.format(i) for i in range(self.scales_before_exp.shape[1])])
        attributes_name.extend(['rot_{}'.format(i) for i in range(self.rotations_before_norm.shape[1])])

        attributes = np.concatenate([
            self.means.clone().detach().cpu().numpy(),
            np.zeros_like(self.means.clone().detach().cpu().numpy()),
            self.res_feats.clone().detach().flatten(start_dim=1).contiguous().cpu().numpy(),
            self.ref_feats.clone().detach().cpu().numpy(),
            self.scales_before_exp.clone().detach().cpu().numpy(),
            self.rotations_before_norm.clone().detach().cpu().numpy()
        ], axis=1)

        # create elements and write to ply file
        data_type = [(name, 'f4') for name in attributes_name]
        elements = np.empty(self.num_anchor_primitive, dtype=data_type)
        elements[:] = list(map(tuple, attributes))
        ply_file = PlyData([PlyElement.describe(elements, 'vertex')])
        ply_file.write(ply_path)

    @torch.no_grad()
    def load_uncompressed(self, ply_path: str) -> None:
        """
        Load uncompressed Gaussian model parameters from ply file.
        """
        assert ply_path.endswith('.ply'), 'Only support ply file'
        # load parameters from ply file
        ply_data = PlyData.read(ply_path).elements[0]

        means = np.stack([np.asarray(ply_data["x"]), np.asarray(ply_data["y"]), np.asarray(ply_data["z"])], axis=1).astype(np.float32)

        scale_names = [p.name for p in ply_data.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((means.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(ply_data[attr_name]).astype(np.float32)

        rot_names = [p.name for p in ply_data.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((means.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(ply_data[attr_name]).astype(np.float32)

        ref_feat_names = [p.name for p in ply_data.properties if p.name.startswith("f_ref_feat_")]
        ref_feat_names = sorted(ref_feat_names, key=lambda x: int(x.split('_')[-1]))
        ref_feats = np.zeros((means.shape[0], len(ref_feat_names)))
        for idx, attr_name in enumerate(ref_feat_names):
            ref_feats[:, idx] = np.asarray(ply_data[attr_name]).astype(np.float32)

        res_feat_names = [p.name for p in ply_data.properties if p.name.startswith("f_res_feat_")]
        res_feat_names = sorted(res_feat_names, key=lambda x: int(x.split('_')[-1]))
        res_feats = np.zeros((means.shape[0], len(res_feat_names)))
        for idx, attr_name in enumerate(res_feat_names):
            res_feats[:, idx] = np.asarray(ply_data[attr_name]).astype(np.float32)
        res_feats = res_feats.reshape((res_feats.shape[0], self.derive_factor, self.res_feats_dim))

        # update parameters
        self.replace_params(param_name='means', param_value=nn.Parameter(torch.tensor(means, dtype=torch.float32, device=self.device), requires_grad=True))
        self.replace_params(param_name='scales_before_exp', param_value=nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=self.device), requires_grad=True))
        self.replace_params(param_name='rotations_before_norm', param_value=nn.Parameter(torch.tensor(rots, dtype=torch.float32, device=self.device), requires_grad=True))
        self.replace_params(param_name='ref_feats', param_value=nn.Parameter(torch.tensor(ref_feats, dtype=torch.float32, device=self.device), requires_grad=True))
        self.replace_params(param_name='res_feats', param_value=nn.Parameter(torch.tensor(res_feats, dtype=torch.float32, device=self.device), requires_grad=True))

    def get_lr_param_pairs(self, lr_configs: dict) -> list[dict]:
        """
        Get learning rate parameter pairs.
        """
        lr_param_pairs = [{
            'params': [getattr(self, param_name), ],
            'lr': lr_configs[cfg_name + '_lr_init'],
            'name': param_name
        } for cfg_name, param_name in self.learnable_param_names.items()]
        return lr_param_pairs

    def get_lr_update_configs(self, lr_configs: dict, spatial_lr_scale: float) -> dict[str, LearningRateUpdateConfig]:
        """
        Get learning rate update configurations.
        """
        lr_update_configs = {
            param_name: LearningRateUpdateConfig(
                lr_init=lr_configs[cfg_name + '_lr_init'] * spatial_lr_scale,
                lr_delay_mult=lr_configs[cfg_name + '_lr_delay_mult'],
                lr_final=lr_configs[cfg_name + '_lr_final'] * spatial_lr_scale
            ) for cfg_name, param_name in self.learnable_param_names.items()
        }
        return lr_update_configs