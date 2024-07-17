import argparse
import random

import numpy as np
import torch

from Modules import TesterCompGS
from Modules.Common import BaseDataset, CustomLogger


# fix random seed
def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    setup_seed(seed=3407)

    parser = argparse.ArgumentParser('\nEval models, if you want to override configurations, please use --key=value')
    parser.add_argument('--dataset_root', type=str, help='root path of dataset', required=True)
    parser.add_argument('--image_folder', type=str, help='image folder', default='images')
    parser.add_argument('--experiment_root', type=str, help='experiment root path', required=True)
    parser.add_argument('--device', type=str, help='device to use', default='cuda')

    args, override_cfgs = parser.parse_known_args()

    dummy_logger = CustomLogger(experiment_dir='Dummy', enable_detail_log=False)
    dataset = BaseDataset(root=args.dataset_root, image_folder=args.image_folder, logger=dummy_logger, device=args.device)

    tester_args = {'device': args.device, 'experiment_root': args.experiment_root, 'dataset': dataset}
    override_cfgs = dict(arg.lstrip('-').split('=') for arg in override_cfgs) if len(override_cfgs) > 0 else {}
    tester_args.update(override_cfgs)

    tester = TesterCompGS(**tester_args)
    tester.eval()
