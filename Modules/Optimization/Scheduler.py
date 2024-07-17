import numpy as np

from .Optimizer import WarpedAdam
from ..Common import LearningRateUpdateConfig, CustomLogger


def exponential_lr_decay(step: int, lr_init: float, lr_final: float, lr_delay_steps: int = 0, lr_delay_mult: float = 1., max_steps: int = 1000000) -> float:
    """
    Calculate current learning rate with exponential decay.
    :param step: current step.
    :param lr_init: initial learning rate.
    :param lr_final: final learning rate.
    :param lr_delay_steps: learning rate delay steps.
    :param lr_delay_mult: learning rate delay multiplier.
    :param max_steps: maximum number of steps.
    :return: adjusted learning rate.
    """
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        return 0.0  # disable this parameter
    delay_rate = 1. if lr_delay_steps <= 0 else lr_delay_mult + (1 - lr_delay_mult) * np.sin(0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
    t = np.clip(step / max_steps, 0, 1)
    lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * lr


def update_lr_means(optimizer: WarpedAdam, logger: CustomLogger, iteration: int, max_iterations: int, lr_update_configs: dict[str, LearningRateUpdateConfig]) -> None:
    """
    Update learning rate of means of Gaussian model.
    :param optimizer: optimizer
    :param logger: logger
    :param iteration: current iteration
    :param max_iterations: maximum number of iterations
    :param lr_update_configs: configurations for learning rate update, including param_name, lr_init, lr_final, lr_delay_steps, lr_delay_mult
    """
    for param_group in optimizer.param_groups:
        if param_group['name'] in lr_update_configs.keys():
            param_name = param_group['name']
            if lr_update_configs[param_name].lr_init == lr_update_configs[param_name].lr_final:
                continue
            updated_lr = exponential_lr_decay(step=iteration, max_steps=max_iterations, lr_init=lr_update_configs[param_name].lr_init,
                                              lr_final=lr_update_configs[param_name].lr_final, lr_delay_mult=lr_update_configs[param_name].lr_delay_mult)
            param_group['lr'] = updated_lr
            logger.add_scalar(tag=f'LearningRate/{param_name}', scalar_value=updated_lr, global_step=iteration)
