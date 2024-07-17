import atexit
import datetime
import logging
import os
import subprocess
from pathlib import Path
from typing import Union

import torch
import yaml
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.tensorboard import SummaryWriter


class CustomLogger:
    def __init__(self, experiment_dir: Union[Path, str], disable_all_log: bool = False, enable_detail_log: bool = True) -> None:
        """
        Custom logger to record information.
        :param experiment_dir: path to the experiment directory.
        :param disable_all_log: whether to disable all logging.
        :param enable_detail_log: whether to enable detail logging.
        """
        if disable_all_log or not os.path.exists(experiment_dir):
            self.enable_log = self.enable_detail_log = False
            return

        self.enable_log = True
        self.enable_detail_log = enable_detail_log

        log_dir = experiment_dir.joinpath('Log/')
        self.text_logger = self.create_text_logger(log_dir=log_dir)

        if self.enable_detail_log:
            tb_dir = experiment_dir.joinpath('Tensorboard/')
            self.tensorboard = self.create_graphical_logger(log_dir=tb_dir)

    def info(self, msg: str, print_: bool = True, force_enable: bool = False) -> None:
        """
        Record information, only record when detailed logging is enabled.
        :param msg: message to be recorded.
        :param print_: whether to print the message.
        :param force_enable: whether to force enable logging.
        """
        if not self.enable_log or (not self.enable_detail_log and not force_enable):
            return
        self.text_logger.info(msg)
        if print_:
            print(msg)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """
        Wrapper to add scalar to tensorboard.
        :param tag: tag of the scalar.
        :param scalar_value: scalar value.
        :param global_step: global step.
        """
        if not self.enable_detail_log:
            return
        self.tensorboard.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, global_step: int) -> None:
        """
        Wrapper to add scalar to tensorboard.
        :param main_tag: main tag of  scalars.
        :param tag_scalar_dict: dictionary of tag and scalar value.
        :param global_step: global step.
        """
        if not self.enable_detail_log:
            return
        self.tensorboard.add_scalars(main_tag=main_tag, tag_scalar_dict=tag_scalar_dict, global_step=global_step)

    def add_histogram(self, tag: str, values: float, global_step: int) -> None:
        """
        Wrapper to add histogram to tensorboard.
        :param tag: tag of the scalar.
        :param values: values.
        :param global_step: global step.
        """
        if not self.enable_detail_log:
            return
        self.tensorboard.add_histogram(tag=tag, values=values, global_step=global_step)

    def add_image(self, tag: str, img_tensor: torch.Tensor, global_step: int) -> None:
        """
        Wrapper to add image to tensorboard, only record when detailed logging is enabled.
        :param tag: tag of the image.
        :param img_tensor: image tensor.
        :param global_step: global step.
        """
        if not self.enable_detail_log:
            return
        self.tensorboard.add_image(tag=tag, img_tensor=img_tensor, global_step=global_step)

    @staticmethod
    def create_text_logger(log_dir: Path) -> logging.Logger:
        """
        Create logger to record information.
        :param log_dir: path to the log directory.
        """
        log_dir.mkdir(exist_ok=True)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(str(log_dir) + '/Log.txt')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def create_graphical_logger(self, log_dir: Path, port: int = 12314) -> SummaryWriter:
        """
        Create logger to record graphical information via tensorboard.
        :param log_dir: path to the log directory.
        :param port: tensorboard port.
        """
        log_dir.mkdir(exist_ok=True)
        logger = SummaryWriter(log_dir=str(log_dir), flush_secs=30)
        self.start_tensorboard(log_dir=str(log_dir), port=port)
        return logger

    @staticmethod
    def start_tensorboard(log_dir: str, port: int) -> None:
        """
        Wrapper to start tensorboard and automatically shutdown when the program unexpectedly exits.
        :param log_dir: tensorboard log directory.
        :param port: tensorboard port.
        """
        cmd = f"tensorboard --logdir={log_dir} --port={port}"
        process = subprocess.Popen(cmd, shell=True)

        def cleanup():
            process.terminate()
            if os.name == 'nt':
                os.system(f'for /f "tokens=5" %a in (\'netstat -ano ^| findstr :{port}\') do if %a NEQ 0 (taskkill /F /PID %a)')
            else:
                os.system(f'pid=$(lsof -t -i:{port}); if [ $pid -ne 0 ]; then kill $pid; fi')
        atexit.register(cleanup)  # shutdown tensorboard when the program unexpectedly exits


def override_value_in_dict(cfg: dict, override_key: str, override_value: str) -> bool:
    """
    Override value in the configuration dictionary.
    :param cfg: configuration dictionary.
    :param override_key: override key.
    :param override_value: override value.
    :return: whether the override is successful.
    """
    for key, value in cfg.items():
        if key == override_key:
            dst_type = type(value)
            cfg[key] = dst_type(override_value) if dst_type != type(override_value) else override_value
            return True
        elif isinstance(value, dict):
            if override_value_in_dict(value, override_key, override_value):
                return True
    return False


def init(config_path: str, override_cfgs: dict) -> tuple[dict, CustomLogger, Path]:
    """
    Initialize configurations and logger.
    :param config_path: configuration yaml filepath
    :param override_cfgs: override configurations passed in command line
    :return configs: configurations dictionary.
    :return logger: logger.
    :return experiment_dir: path to the experiment directory.
    """
    with open(config_path, mode='r') as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)
    # override configurations
    for key, value in override_cfgs.items():
        override_state = override_value_in_dict(cfg=configs, override_key=key, override_value=value)
        assert override_state, f'Key {key} does not exist in the configuration file {config_path}'

    experiment_dir = Path(configs['training']['save_directory'])
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = Path(str(experiment_dir) + '/' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
    experiment_dir.mkdir(exist_ok=True)

    enable_detail_log = not configs['training']['disable_logging'] if 'disable_logging' in configs['training'] else True
    logger = CustomLogger(experiment_dir=experiment_dir, enable_detail_log=enable_detail_log)

    # log environment information
    env_info = collect_env_info(configs=configs)
    logger.info(env_info, print_=False, force_enable=True)

    return configs, logger, experiment_dir


def collect_env_info(configs: dict) -> str:
    """
    Collect environment information.
    """
    env_info = '===================== Environment Information =====================\n\n'
    env_info += get_pretty_env_info()
    env_info += '\n\n\nThird-party Pip libraries:\n\n'
    env_info += os.popen('pip list').read()
    env_info += '\n\n===================== Current Branch ===================== \n\n'
    try:
        env_info += [branch.lstrip('* ') for branch in os.popen('git branch').read().split('\n') if '*' in branch][0]
    except:
        env_info += 'UNKNOWN'
    env_info += '\n\n===================== Configuration ===================== \n\n'
    env_info += yaml.dump(configs)
    return env_info
