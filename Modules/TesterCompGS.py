import json
import os
import time

import lpips
import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ssim
from torch.nn import functional as F
from tqdm import tqdm

from Modules.Common import BaseDataset, CustomLogger, Sample, RenderSettings
from Modules.GaussianModels import GaussianModel
from Modules.TrainerCompGS import TrainerCompGS


class TesterCompGS:
    def __init__(self, trainer: TrainerCompGS = None, experiment_root: str = None, device: str = None, dataset: BaseDataset = None, **kwargs) -> None:
        if trainer is not None:  # inherit configurations from trainer
            self.device = trainer.device
            self.experiment_root = trainer.experiment_dir
            self.eval_dataset = trainer.dataset
            self.gaussian_model = trainer.gaussian_model
            self.gpcc_codec_path = trainer.gpcc_codec_path
        else:  # use custom configurations
            self.device = device
            self.experiment_root = experiment_root
            self.eval_dataset = dataset
            self.gaussian_model = self.init_gaussian_model(kwargs)
            self.gpcc_codec_path = kwargs['gpcc_codec_path']

        # load Gaussian model
        decompression_time = self.load_gaussian_model()  # minutes

        # load LPIPS model for evaluation
        self.lpips_model = lpips.LPIPS(net='vgg', version='0.1', verbose=False).eval().to(self.device)

        # create folder to store original and rendered images
        self.eval_results_folder = os.path.join(self.experiment_root, 'eval')
        os.makedirs(self.eval_results_folder, exist_ok=True)
        self.original_img_folder = os.path.join(self.eval_results_folder, 'original')
        os.makedirs(self.original_img_folder, exist_ok=True)
        self.rendered_img_folder = os.path.join(self.eval_results_folder, 'rendered')
        os.makedirs(self.rendered_img_folder, exist_ok=True)

        self.records = {'decompression_time': decompression_time}

    @torch.no_grad()
    def eval(self) -> None:
        assert self.gaussian_model.init_params, 'Gaussian model must be initialized before evaluation.'
        self.gaussian_model.eval()
        self.eval_dataset.eval()

        per_view_record = {}
        total_predict_time = total_render_time = 0
        # iterate over all test views
        for view_idx in tqdm(range(len(self.eval_dataset)), ncols=50):
            # load image and camera calibration matrix
            sample = self.eval_dataset[view_idx]

            # rendering
            rendered_img, predict_time, render_time = self.render(sample)

            total_predict_time += predict_time
            total_render_time += render_time

            # save original and rendered image
            original_img_path = os.path.join(self.original_img_folder, f'{view_idx:04d}.png')
            original_img = Image.fromarray((sample.img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            original_img.save(original_img_path)

            rendered_img_path = os.path.join(self.rendered_img_folder, f'{view_idx:04d}.png')
            rendered_img = Image.fromarray((rendered_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            rendered_img.save(rendered_img_path)

            # calculate quality scores
            quality_scores = self.calculate_quality_scores(original_img_path=original_img_path, rendered_img_path=rendered_img_path)
            per_view_record[f'view_{view_idx:04d}'] = quality_scores

        # calculate average quality scores
        avg_recoder = {'PSNR': sum([per_view_record[key]['PSNR'] for key in per_view_record]) / len(per_view_record),
                       'SSIM': sum([per_view_record[key]['SSIM'] for key in per_view_record]) / len(per_view_record),
                       'LPIPS': sum([per_view_record[key]['LPIPS'] for key in per_view_record]) / len(per_view_record)}

        # save evaluation results
        self.records.update({'per_view': per_view_record, 'average': avg_recoder, 'render_time': total_render_time / len(self.eval_dataset)})
        self.records['decompression_time'] = self.records['decompression_time'] + total_predict_time / 60  # minutes

        # save evaluation results to json file, including:
        if os.path.exists(os.path.join(self.eval_results_folder, 'results.json')):  # some training results should be already saved
            with open(os.path.join(self.eval_results_folder, 'results.json'), 'r') as f:
                already_saved = json.load(f)
            self.records.update({key: value for key, value in already_saved.items() if key not in self.records})
        else:
            print('Warning: no training results found.')
        with open(os.path.join(self.eval_results_folder, 'results.json'), 'w') as f:
            json.dump(self.records, f, indent=4)

    def init_gaussian_model(self, configs: dict) -> GaussianModel:
        """
        Init Gaussian model from configurations.
        """
        gaussian_model = GaussianModel(
            logger=CustomLogger(experiment_dir='Dummy', enable_detail_log=False), device=self.device,
            voxel_size=float(configs['voxel_size']), derive_factor=int(configs['derive_factor']),
            ref_feats_dim=int(configs['ref_feats_dim']), ref_hyper_dim=int(configs['ref_hyper_dim']),
            res_feats_dim=int(configs['res_feats_dim']), res_hyper_dim=int(configs['res_hyper_dim'])
        ).to(self.device)

        return gaussian_model

    def load_gaussian_model(self) -> float:
        """
        Load Gaussian model.
        """
        # first load weights of network
        weights_path = os.path.join(self.experiment_root, 'point_cloud', 'weights.pth')
        self.gaussian_model.load_weights(weight_path=weights_path)

        # then load bitstreams
        npz_path = os.path.join(self.experiment_root, 'point_cloud', 'bitstreams.npz')
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        self.gaussian_model.load_compressed_params(npz_path=npz_path, gpcc_codec_path=self.gpcc_codec_path)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        decompression_time = (end_time - start_time) / 60  # minutes

        return decompression_time

    @torch.no_grad()
    def render(self, sample: Sample) -> tuple[torch.Tensor, float, float]:
        """
        Render a view.
        :param sample: sample with camera calibration matrix.
        :return rendered_img: rendered image.
        :return predict_time: time for prediction in seconds.
        :return render_time: time for rendering in milliseconds.
        """
        render_settings = RenderSettings(
            image_height=sample.image_height, image_width=sample.image_width,
            tanfovx=sample.tan_half_fov_x, tanfovy=sample.tan_half_fov_y, campos=sample.camera_center,
            cam_idx=sample.cam_idx,
            viewmatrix=sample.world_to_view_proj_mat, projmatrix=sample.world_to_image_proj_mat)

        rendered_img, predict_time, render_time = self.gaussian_model.render_inference(render_settings=render_settings)
        return rendered_img, predict_time, render_time

    @torch.no_grad()
    def calculate_quality_scores(self, original_img_path: str, rendered_img_path: str) -> dict:
        # load original and rendered images
        original_img, rendered_img = Image.open(original_img_path), Image.open(rendered_img_path)

        # calculate rendering PSNR
        original_img = torch.from_numpy(np.array(original_img)).permute(2, 0, 1).float().unsqueeze(dim=0) / 255.
        rendered_img = torch.from_numpy(np.array(rendered_img)).permute(2, 0, 1).float().unsqueeze(dim=0) / 255.
        rendered_mse = F.mse_loss(original_img, rendered_img)
        psnr = 10 * torch.log10(1. / rendered_mse).item()

        # calculate rendering SSIM score
        ssim_score = ssim(original_img, rendered_img, data_range=1., size_average=True).item()

        # calculate rendering LPIPS score
        lpips_score = self.lpips_model(original_img.to(self.device), rendered_img.to(self.device)).item()

        return {'PSNR': psnr, 'SSIM': ssim_score, 'LPIPS': lpips_score}