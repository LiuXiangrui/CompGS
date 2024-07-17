import json
import os
import time
from functools import partial

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ssim
from torch.nn import functional as F
from tqdm import tqdm

from Modules.Common import BaseDataset, RenderSettings, RenderResults, Sample, init
from Modules.GaussianModels import GaussianModel
from Modules.Optimization import AdaptiveControl, WarpedAdam, update_lr_means


class TrainerCompGS:
    def __init__(self, config_path: str, override_cfgs: dict) -> None:
        """
        :param config_path: configuration yaml filepath
        :param override_cfgs: override configurations passed in command line
        """
        self.configs, self.logger, self.experiment_dir = init(config_path=config_path, override_cfgs=override_cfgs)
        self.logger.info(f'\nTrainer: {type(self).__name__}\n')

        self.checkpoints_dir = self.experiment_dir.joinpath("point_cloud/")
        self.checkpoints_dir.mkdir(exist_ok=True)

        self.device = 'cuda' if self.configs['training']['gpu'] else 'cpu'

        # init train / eval dataset
        self.dataset = BaseDataset(root=self.configs['dataset']['root'], image_folder=self.configs['dataset']['image_folder'], logger=self.logger, device=self.device)

        # init Gaussian model
        self.gaussian_model = GaussianModel(
            logger=self.logger, device=self.device,
            voxel_size=self.configs['gaussians']['voxel_size'], derive_factor=self.configs['gaussians']['derive_factor'],
            ref_feats_dim=self.configs['gaussians']['ref_feats_dim'], ref_hyper_dim=self.configs['gaussians']['ref_hyper_dim'],
            res_feats_dim=self.configs['gaussians']['res_feats_dim'], res_hyper_dim=self.configs['gaussians']['res_hyper_dim'],
        )

        # init gaussian model with sparse point clouds
        self.gaussian_model.init_from_sfm(sfm_point_cloud_path=self.dataset.sfm_point_cloud_path)

        # init optimizer
        self.gaussian_optimizer, self.aux_optimizer = self.init_optimizers()

        # init lr scheduler
        self.gaussian_lr_scheduler = self.init_lr_scheduler()

        self.adaptive_control_function = AdaptiveControl(
            logger=self.logger, gaussian_model=self.gaussian_model, optimizer=self.gaussian_optimizer,
            couple_threshold=self.configs['adaptive_control']['couple_threshold'], grad_threshold=self.configs['adaptive_control']['grad_threshold'],
            opacity_threshold=self.configs['adaptive_control']['opacity_threshold'], update_depth=self.configs['adaptive_control']['update_depth'],
            update_hierarchy_factor=self.configs['adaptive_control']['update_hierarchy_factor'], update_init_factor=self.configs['adaptive_control']['update_init_factor'],
        )

        self.gpcc_codec_path = self.configs['training']['gpcc_codec_path']

    def train(self) -> None:
        """
        Train Gaussian model.
        """
        self.gaussian_model.train()
        self.dataset.train()
        start_iteration = self.load_checkpoint()

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for iteration in tqdm(range(start_iteration, self.configs['training']['max_iterations'] + 1), ncols=50):
            # randomly select a view to render
            sample = self.dataset[0]

            # learning rate decay
            self.gaussian_lr_scheduler(iteration=iteration)

            # forward
            render_settings = RenderSettings(
                cam_idx=sample.cam_idx, image_height=sample.image_height, image_width=sample.image_width,
                tanfovx=sample.tan_half_fov_x, tanfovy=sample.tan_half_fov_y, campos=sample.camera_center,
                viewmatrix=sample.world_to_view_proj_mat, projmatrix=sample.world_to_image_proj_mat)

            retain_grad = iteration < self.configs['adaptive_control']['stop_iteration']
            render_results = self.gaussian_model.render(render_settings=render_settings, retain_grad=retain_grad)

            # backward
            backward_results = self.backward(iteration=iteration, sample=sample, render_results=render_results)

            # record
            self.record(iteration=iteration, backward_results=backward_results, render_results=render_results, sample=sample)

            # optimize Gaussian parameters
            self.optimize(iteration=iteration, render_results=render_results)

            # save checkpoints
            self.save_checkpoints(iteration=iteration)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        training_time = (end_time - start_time) / 60  # minutes

        # evaluate the model
        self.eval()

        # compress the model and save weights
        compression_results = self.compress_gaussians()

        # save results to json file
        self.save_results(training_time=training_time, compression_results=compression_results)

    @torch.no_grad()
    def eval(self) -> None:
        """
        Evaluate Gaussian model without actual compression.
        """
        self.gaussian_model.eval()
        self.dataset.eval()

        save_dir = os.path.join(self.experiment_dir, 'eval_training')
        os.makedirs(save_dir, exist_ok=True)
        original_img_folder = os.path.join(save_dir, 'original')
        os.makedirs(original_img_folder, exist_ok=True)
        rendered_img_folder = os.path.join(save_dir, 'rendered')
        os.makedirs(rendered_img_folder, exist_ok=True)

        per_view_record, eval_time = {}, 0
        for view_idx in tqdm(range(len(self.dataset)), ncols=50):
            # load image and camera calibration matrix
            sample = self.dataset[view_idx]

            # rendering
            torch.cuda.synchronize()
            start_time = time.perf_counter() * 1000
            render_settings = RenderSettings(
                cam_idx=sample.cam_idx, image_height=sample.image_height, image_width=sample.image_width,
                tanfovx=sample.tan_half_fov_x, tanfovy=sample.tan_half_fov_y, campos=sample.camera_center,
                viewmatrix=sample.world_to_view_proj_mat, projmatrix=sample.world_to_image_proj_mat)
            render_results = self.gaussian_model.render(render_settings=render_settings)
            torch.cuda.synchronize()
            end_time = time.perf_counter() * 1000
            eval_time += end_time - start_time

            # save original and rendered image
            original_img_path = os.path.join(original_img_folder, f'{view_idx}.png')
            original_img = Image.fromarray((sample.img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            original_img.save(original_img_path)

            # save rendered image
            rendered_img = render_results.rendered_img
            rendered_img_path = os.path.join(rendered_img_folder, f'{view_idx}.png')
            rendered_img = Image.fromarray((rendered_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            rendered_img.save(rendered_img_path)

            # load rendered image and calculate PSNR
            original_img = Image.open(original_img_path)
            rendered_img = Image.open(rendered_img_path)
            original_img = torch.tensor(np.array(original_img)).permute(2, 0, 1).float() / 255.
            rendered_img = torch.tensor(np.array(rendered_img)).permute(2, 0, 1).float() / 255.

            mse = F.mse_loss(original_img, rendered_img)
            psnr = 10 * torch.log10(1 / mse)

            per_view_record[f'view_{view_idx}'] = psnr.item()

        results = {'per_view': per_view_record, 'test_time': eval_time, 'average': sum([per_view_record[key] for key in per_view_record]) / len(per_view_record)}

        # save evaluation results
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=True)

    def init_optimizers(self) -> tuple[WarpedAdam, WarpedAdam]:
        """
        Initialize main and auxiliary optimizers.
        """
        assert self.gaussian_model.init_params, 'Gaussian model must be initialized before initializing optimizer.'
        lr_config_groups = self.configs['training']['lr']

        gaussian_lr_pairs = self.gaussian_model.gaussian_params.get_lr_param_pairs(lr_configs=lr_config_groups['gaussian'])
        network_lr_pairs = self.gaussian_model.network.get_lr_param_pairs(lr_config_groups=lr_config_groups)
        params_lr_pairs = gaussian_lr_pairs + network_lr_pairs

        gaussian_optimizer = WarpedAdam(params_lr_pairs, lr=0., eps=1e-15)

        aux_optimizer = WarpedAdam(self.gaussian_model.network.get_lr_aux_param_pairs(), lr=0., eps=1e-15)

        return gaussian_optimizer, aux_optimizer

    def init_lr_scheduler(self) -> partial:
        """
        Initialize preprocess functions.
        """
        # initialize lr scheduler for means of Gaussians.
        lr_config_groups = self.configs['training']['lr']
        spatial_lr_scale = self.dataset.screen_extent

        gaussian_params_lr_update_configs = self.gaussian_model.gaussian_params.get_lr_update_configs(lr_configs=lr_config_groups['gaussian'], spatial_lr_scale=spatial_lr_scale)
        network_params_lr_update_configs = self.gaussian_model.network.get_lr_update_configs(lr_config_groups=lr_config_groups)
        lr_update_configs = {**gaussian_params_lr_update_configs, **network_params_lr_update_configs}

        gaussian_lr_scheduler = partial(update_lr_means,
                                        optimizer=self.gaussian_optimizer, logger=self.logger,
                                        lr_update_configs=lr_update_configs, max_iterations=self.configs['training']['max_iterations'])

        return gaussian_lr_scheduler

    def backward(self, iteration: int, sample: Sample, render_results: RenderResults) -> dict:
        """
        Backward pass, return backward records for logging.
        :param iteration: current iteration.
        :param sample: training sample.
        :param render_results: rendering results.
        """
        # calculate rendering loss
        ssim_weight = self.configs['training']['ssim_weight']
        l1_loss = F.l1_loss(render_results.rendered_img, sample.img)
        ssim_loss = 1 - ssim(render_results.rendered_img.unsqueeze(dim=0), sample.img.unsqueeze(dim=0), data_range=1., size_average=True)
        rendering_loss = (1 - ssim_weight) * l1_loss + ssim_weight * ssim_loss

        # calculate regularization loss
        reg_loss = 0.01 * render_results.scales.prod(dim=1).mean()

        # calculate rate loss
        bpp = render_results.bpp
        rate_loss = self.configs['training']['lambda_weight'] * (sum(v for v in bpp.values()))

        if iteration == self.configs['training']['rate_loss_start_iteration']:
            self.logger.info(f'Start optimizing rate loss from iteration {iteration}...')

        loss = rendering_loss + reg_loss + (rate_loss if iteration > self.configs['training']['rate_loss_start_iteration'] else 0.)

        # backward
        loss.backward()

        # calculate auxiliary loss
        aux_loss = self.gaussian_model.aux_loss
        aux_loss.backward()

        # return loss for logging
        return {
            'loss': {'rendering': rendering_loss.item(), 'reg': reg_loss.item(), 'rate': rate_loss.item(), 'total': loss.item()},
            'aux_loss': {'total': aux_loss.item()},
            'rendering': {'l1': l1_loss.item(), 'ssim': ssim_loss.item()},
            'bpp': {k: v.item() for k, v in bpp.items()}
        }

    def record(self, iteration: int, sample: Sample, render_results: RenderResults, backward_results: dict) -> None:
        """
        Record information for logging.
        :param iteration: current iteration.
        :param sample: training sample.
        :param render_results: rendering results.
        :param backward_results: backward results.
        """
        # record Gaussian information to tensorboard
        self.logger.add_scalar(tag='gaussians/num_anchor_primitive', scalar_value=self.gaussian_model.num_anchor_primitive, global_step=iteration)

        # record loss to tensorboard
        for main_tag in backward_results:
            self.logger.add_scalars(main_tag=main_tag, tag_scalar_dict=backward_results[main_tag], global_step=iteration)

        # record rendered image and original image to tensorboard
        if iteration != 0 and iteration % 5000 == 0:
            self.logger.add_image(tag='images/original', img_tensor=sample.img, global_step=iteration)
            self.logger.add_image(tag='images/rendered', img_tensor=render_results.rendered_img, global_step=iteration)

    def save_checkpoints(self, iteration: int) -> None:
        """
        Save uncompressed Gaussians parameters and model weights to checkpoints folder.
        :param iteration: current iteration.
        """
        if iteration == 0 or iteration % self.configs['training']['save_interval'] != 0:
            return
        ckpt_folder = os.path.join(self.checkpoints_dir, f'iteration_{iteration}')
        os.makedirs(ckpt_folder, exist_ok=True)
        self.gaussian_model.save_uncompressed_params(os.path.join(ckpt_folder, 'point_cloud.ply'))
        self.gaussian_model.save_weights(os.path.join(ckpt_folder, 'weights.pth'))

    def optimize(self, iteration: int, render_results: RenderResults) -> None:
        """
        Optimize parameters.
        :param iteration: current iteration.
        :param render_results: rendering results used for adaptive control.
        """
        num_training_views = len(self.dataset)

        # update rendering information for adaptive control
        aux_update_enable = self.configs['adaptive_control']['update_aux_start_base_interval'] * num_training_views < iteration < self.configs['adaptive_control']['stop_iteration']
        if aux_update_enable:
            self.gaussian_model.update_aux_params(render_results=render_results)

        # release memory of auxiliary variables after adaptive control
        if iteration == self.configs['adaptive_control']['stop_iteration']:
            self.gaussian_model.remove_aux_params()
 
        # optimize Gaussian parameters by gradient descent
        self.gaussian_optimizer.step()
        self.gaussian_optimizer.zero_grad(set_to_none=True)

        self.aux_optimizer.step()
        self.aux_optimizer.zero_grad(set_to_none=True)

        # adaptive control
        adaptive_control_enable = (self.configs['adaptive_control']['control_start_base_interval'] * num_training_views < iteration < self.configs['adaptive_control']['stop_iteration']
                                   and iteration % (self.configs['adaptive_control']['control_base_interval'] * num_training_views) == 0)
        if adaptive_control_enable:
            self.adaptive_control_function.control()

        if iteration % 1000 == 0:
            torch.cuda.empty_cache()

    def load_checkpoint(self) -> int:
        """
        Load checkpoint for Gaussian model and return start iteration.
        """
        if 'checkpoint_dir' not in self.configs['training']:
            return 0
        checkpoint_path = os.path.join(self.configs['training']['checkpoint_dir'], 'point_cloud.ply')
        weight_path = os.path.join(self.configs['training']['checkpoint_dir'], 'weights.pth')
        last_iteration = os.path.splitext(os.path.split(self.configs['training']['checkpoint_dir'])[-1])[0].split('_')[-1]  # start iteration
        self.gaussian_model.load_uncompressed_params(ply_path=checkpoint_path)
        self.gaussian_model.load_weights(weight_path=weight_path)
        start_iteration = int(last_iteration) + 1
        self.logger.info(f'Load checkpoint from {self.configs["training"]["checkpoint_dir"]}...')
        return start_iteration

    @torch.no_grad()
    def compress_gaussians(self) -> dict:
        """
        compress Gaussians model.
        """
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # save network weights
        weights_path = os.path.join(self.checkpoints_dir, 'weights.pth')
        self.gaussian_model.save_weights(weights_path, update_entropy_model=True)

        # compress the model
        bin_path = os.path.join(self.checkpoints_dir, 'bitstreams.npz')
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        size_results = self.gaussian_model.save_compressed_params(npz_path=bin_path, gpcc_codec_path=self.gpcc_codec_path)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        compression_time = (end_time - start_time) / 60  # minutes

        weights_size = os.path.getsize(weights_path) / 1024 / 1024  # MB
        size_results['weights_size'] = weights_size
        size_results['total'] = size_results['total'] + weights_size

        compression_results = {
            'file_size': size_results,
            'compression_time': compression_time
        }

        return compression_results

    def save_results(self, training_time: float, compression_results: dict) -> None:
        """
        Save model size, training time and number of gaussians to result json file.
        :param training_time: training time in minutes.
        :param compression_results: results of model size.
        """
        eval_results_folder = os.path.join(self.experiment_dir, 'eval')
        os.makedirs(eval_results_folder, exist_ok=True)

        results = {
            'num_gaussians': self.gaussian_model.num_anchor_primitive,
            'training_time': training_time  # minutes
        }

        results.update(compression_results)

        with open(os.path.join(eval_results_folder, 'results.json'), 'w') as f:
            json.dump(results, f, indent=True)