from functools import reduce

import torch
from einops import repeat
from torch_scatter import scatter_max

from .Optimizer import WarpedAdam
from ..Common import CustomLogger
from ..GaussianModels import GaussianModel


class AdaptiveControl:
    def __init__(self, logger: CustomLogger, gaussian_model: GaussianModel, optimizer: WarpedAdam,
                 couple_threshold: int, grad_threshold: float, opacity_threshold: float,
                 update_depth: int, update_init_factor: int, update_hierarchy_factor: int) -> None:
        """
        Adaptive control of Gaussians.
        :param logger: logger.
        :param gaussian_model: Gaussian model.
        :param optimizer: optimizer.
        :param couple_threshold: threshold to control the success of the model.
        :param grad_threshold: Gaussians with accumulated gradient of means larger than grad_threshold * extent will be densified.
        :param opacity_threshold: Gaussians with opacity lower than opacity_threshold will be removed.
        :param update_depth: depth of update.
        :param update_init_factor: factor of initial update.
        :param update_hierarchy_factor: factor of hierarchical update.
        """
        self.logger, self.gaussian_model, self.optimizer = logger, gaussian_model, optimizer
        self.couple_threshold, self.grad_threshold, self.opacity_threshold = couple_threshold, grad_threshold, opacity_threshold
        self.update_depth, self.update_init_factor, self.update_hierarchy_factor = update_depth, update_init_factor, update_hierarchy_factor

    @torch.no_grad()
    def control(self) -> None:
        self.growing()
        self.prune()
        self.gaussian_model.reset_aux_params()

    @torch.no_grad()
    def growing(self):
        # calculate average gradients
        grads = self.gaussian_model.accumulated_grads / self.gaussian_model.coupled_denorm
        grads[grads.isnan()] = 0.
        grads = torch.norm(grads, dim=-1)

        # only coupled primitive with accessed times larger than threshold will be densified
        candidate_coupled_primitive_mask = torch.gt(self.gaussian_model.coupled_denorm, self.couple_threshold).squeeze(dim=1)

        num_coupled_primitive_before_growing = self.gaussian_model.num_coupled_primitive
        for level in range(self.update_depth):
            # update threshold, where accumulated gradients than the threshold will be added to current level
            cur_threshold = self.grad_threshold * ((self.update_hierarchy_factor // 2) ** level)

            # voxels with gradients larger than threshold are deemed as significant
            candidate_mask = torch.logical_and(torch.ge(grads, cur_threshold), candidate_coupled_primitive_mask)

            # random pick candidates to retain
            rand_mask = torch.gt(torch.rand_like(candidate_mask.float()), 0.5 ** (level + 1))
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            # calculate number of previously added primitives
            num_added_coupled_primitive = self.gaussian_model.num_coupled_primitive - num_coupled_primitive_before_growing

            if level > 0 and num_added_coupled_primitive == 0:  # no anchor primitive is added in the previous level
                continue
            elif level > 0 and num_added_coupled_primitive != 0:  # extend candidate mask to be consistent with the number of coupled primitives
                candidate_mask = torch.cat([candidate_mask, torch.zeros(num_added_coupled_primitive, dtype=torch.bool, device=self.gaussian_model.device)], dim=0)

            # voxelize anchor primitives at current level
            voxel_size = self.gaussian_model.voxel_size * (self.update_init_factor // (self.update_hierarchy_factor ** level))
            grid_coords = torch.round(self.gaussian_model.means / voxel_size).int()

            # voxelize candidate coupled primitives
            candidate_grid_coords = torch.round(self.gaussian_model.pred_gaussian_means.view(-1, 3)[candidate_mask] / voxel_size).int()
            candidate_grid_coords_unique, inverse_indices = torch.unique(candidate_grid_coords, return_inverse=True, dim=0)

            # remove candidates whose grid coordinates are already occupied by anchor primitives
            chunk_size = 4096
            num_chunks = grid_coords.shape[0] // chunk_size + int(grid_coords.shape[0] % chunk_size != 0)
            remove_duplicates_list = []
            for chunk_id in range(num_chunks):
                chunk_grid_coords = grid_coords[chunk_id * chunk_size: (chunk_id + 1) * chunk_size, :]
                cur_remove_duplicates = torch.eq(candidate_grid_coords_unique.unsqueeze(dim=1), chunk_grid_coords).all(dim=-1).any(dim=-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)

            # generate new anchor primitives
            retain_mask = ~remove_duplicates
            new_means = candidate_grid_coords_unique[retain_mask] * voxel_size
            if new_means.numel() == 0:  # no new primitives are added
                continue

            new_scaling = torch.log(torch.ones_like(new_means).repeat(1, 2) * voxel_size)  # [N, 6]
            new_rotation = torch.zeros((new_means.shape[0], 4), device=new_means.device)  # [N, 4]
            new_rotation[:, 0] = 1.

            new_ref_feats = repeat(self.gaussian_model.ref_feats, 'n c -> (n k) c', k=self.gaussian_model.derive_factor)[candidate_mask]
            new_ref_feats = scatter_max(new_ref_feats, index=repeat(inverse_indices, 'n -> n c', c=new_ref_feats.shape[-1]), dim=0)[0][retain_mask]

            new_res_feats = torch.zeros(new_means.shape[0], self.gaussian_model.derive_factor, self.gaussian_model.res_feats_dim, device=self.gaussian_model.device)

            new_anchor_primitives = dict(means=new_means, scales_before_exp=new_scaling, rotations_before_norm=new_rotation, ref_feats=new_ref_feats, res_feats=new_res_feats)

            # add new anchor primitives
            for param_name, param_values in new_anchor_primitives.items():
                extent_params = self.optimizer.extend_params(name=param_name, values=param_values)
                self.gaussian_model.replace_params(param_name=param_name, param_value=extent_params)

        # update auxiliary parameters for the subsequent pruning
        num_added_coupled_primitive = self.gaussian_model.num_coupled_primitive - num_coupled_primitive_before_growing
        num_added_anchor_primitive = num_added_coupled_primitive // self.gaussian_model.derive_factor

        extended_anchor_denorm = torch.cat([self.gaussian_model.anchor_denorm, torch.zeros(num_added_anchor_primitive, 1, device=self.gaussian_model.device, requires_grad=False)], dim=0)
        self.gaussian_model.replace_aux_params(param_name='anchor_denorm', param_value=extended_anchor_denorm)

        extended_accumulated_opacities = torch.cat([self.gaussian_model.accumulated_opacities, torch.zeros(num_added_anchor_primitive, 1, device=self.gaussian_model.device, requires_grad=False)], dim=0)
        self.gaussian_model.replace_aux_params(param_name='accumulated_opacities', param_value=extended_accumulated_opacities)

        self.logger.info(f'Adaptive control growing: add {num_added_anchor_primitive} anchor primitives.')

    @torch.no_grad()
    def prune(self) -> None:
        previous_num_anchor_primitives = self.gaussian_model.num_anchor_primitive
        prune_mask = torch.lt(self.gaussian_model.accumulated_opacities, self.opacity_threshold * self.gaussian_model.anchor_denorm).squeeze(dim=1)
        retained_mask = ~prune_mask
        assert retained_mask.shape[0] == self.gaussian_model.num_anchor_primitive, f"Retained mask shape {retained_mask.shape[0]} not equal to number of anchor primitives {self.gaussian_model.num_anchor_primitive}."

        # remove anchor primitives with opacity lower than threshold
        retained_params = self.optimizer.remove_params_by_mask(retained_mask=retained_mask)
        for param_name, param_value in retained_params.items():
            self.gaussian_model.replace_params(param_name=param_name, param_value=param_value)

        self.logger.info(f'Adaptive control pruning: remove {previous_num_anchor_primitives - self.gaussian_model.num_anchor_primitive} anchor primitives.')