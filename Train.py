import argparse
import random

import numpy as np
import torch

from Modules import TrainerCompGS, TesterCompGS


# fix random seed
def setup_seed(seed: int) -> None:
    """
    Fix random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    setup_seed(seed=3407)

    parser = argparse.ArgumentParser('\nTrain and eval models, if you want to override configurations, please use --key=value')
    parser.add_argument('--config', type=str, help='filepath of configuration files', required=True)
    args, override_cfgs = parser.parse_known_args()
    override_cfgs = dict(arg.lstrip('-').split('=') for arg in override_cfgs) if len(override_cfgs) > 0 else {}

    # train Gaussian model
    trainer = TrainerCompGS(config_path=args.config, override_cfgs=override_cfgs)
    trainer.train()

    # test Gaussian model
    tester = TesterCompGS(trainer=trainer)
    tester.eval()
