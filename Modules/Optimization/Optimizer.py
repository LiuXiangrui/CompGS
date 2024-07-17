import torch
import torch.nn as nn
from torch.optim import Adam


class WarpedAdam(Adam):
    """
    Warped Adam optimizer, adding helper functions to support params update.
    """
    def __init__(self, params, lr: float = 0., betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        super().__init__(params, lr=lr, betas=betas, eps=eps)

    def extend_params(self, name: str, values: torch.Tensor) -> nn.Parameter:
        """
        Extend optimized parameters.
        :param name: name of parameters to be extent.
        :param values: values to be added to the parameters.
        :return: extent parameters.
        """
        for group in self.param_groups:
            if group['name'] == name:
                assert len(group['params']) == 1, f"Length of params in group {name} must be 1, but got {len(group['params'])}."
                # obtain the state of params need to be extended
                stored_state = self.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state['exp_avg'] = torch.cat([stored_state['exp_avg'], torch.zeros_like(values)], dim=0)
                    stored_state['exp_avg_sq'] = torch.cat([stored_state['exp_avg_sq'], torch.zeros_like(values)], dim=0)
                    del self.state[group['params'][0]]
                    # update params need to be extended
                    group['params'][0] = nn.Parameter(torch.cat([group['params'][0], values], dim=0), requires_grad=True)
                    # update optimizer state
                    self.state[group['params'][0]] = stored_state
                else:
                    group['params'][0] = nn.Parameter(torch.cat([group['params'][0], values], dim=0), requires_grad=True)
                # return extent params to update Gaussian model
                updated_params = group['params'][0]
                return updated_params
        assert False, f"Params {name} to be extended not found in optimizer."

    def remove_params_by_mask(self, retained_mask: torch.Tensor) -> dict:
        """
        Remove elements of parameters in optimizer by mask.
        :param retained_mask: mask of parameters to be retained.
        :return: retained parameters.
        """
        assert retained_mask.dim() == 1, f"Dimension of retained_mask must be 1, but got {retained_mask.dim()}."
        retained_params = {}
        for group in self.param_groups:
            if len(group['params']) > 1 or group['params'][0].shape[0] != retained_mask.shape[0]:  # exclude network parameters
                continue
            # obtain the state of params need to be removed
            stored_state = self.state.get(group['params'][0], None)
            # update optimizer state
            if stored_state is not None:
                stored_state['exp_avg'] = stored_state['exp_avg'][retained_mask]
                stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][retained_mask]
                del self.state[group['params'][0]]
                group['params'][0] = nn.Parameter(group['params'][0][retained_mask], requires_grad=True)
                self.state[group['params'][0]] = stored_state
            else:
                # remove params indicated by mask
                group['params'][0] = nn.Parameter(group['params'][0][retained_mask], requires_grad=True)
            # clamp scaling vectors for scales according to the official codes of ScaffoldGS
            if group['name'] == 'scales_before_exp':
                scaling_factors = group['params'][0][:, 3:]
                scaling_factors[scaling_factors > 0.05] = 0.05
                group['params'][0][:, 3:] = scaling_factors
            retained_params[group['name']] = group['params'][0]
        return retained_params
