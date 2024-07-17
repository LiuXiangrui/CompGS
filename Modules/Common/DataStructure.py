from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GaussianParameters:
    means: torch.Tensor  # means, shape (N, 3)
    scales: torch.Tensor  # scaling vectors, shape (N, 3)
    rotations: torch.Tensor  # rotation quaternions, shape (N, 4)
    opacities: torch.Tensor  # opacities, shape (N,)
    colors: torch.Tensor  # view-dependent colors, shape (N, 3)


@dataclass
class RenderSettings:
    image_height: int  # image height
    image_width: int  # image width
    tanfovx: float  # tan fov x
    tanfovy: float  # tan fov y
    viewmatrix: torch.Tensor  # projection matrix from world space to camera space, shape (4, 4)
    projmatrix: torch.Tensor  # projection matrix from world space to image plane, shape (4, 4)
    cam_idx: int  # camera id
    campos: torch.Tensor  # locations of camera center in world space, shape (3,)


@dataclass
class LearningRateUpdateConfig:
    lr_init: float  # initial learning rate
    lr_delay_mult: float  # learning rate delay multiplier
    lr_final: float  # final learning rate


@dataclass
class DatasetSample:
    img: torch.Tensor  # image, shape (3, H, W)
    img_height: int  # image height
    img_width: int  # image width
    img_name: str  # image name

    cam_idx: int  # camera id
    camera_center: torch.Tensor  # camera center, shape (3,)

    fov_x: float  # field of view in horizontal direction
    fov_y: float  # field of view in vertical direction
    tan_half_fov_x: float  # tangent of half of the field of view in horizontal direction
    tan_half_fov_y: float  # tangent of half of the field of view in vertical direction

    rotation_mat: np.ndarray  # rotation matrix, shape (3, 3)
    translation_vec: np.ndarray  # translation vector, shape (3,)

    world_to_view_proj_mat: torch.Tensor  # world to view projection matrix, shape (4, 4)
    world_to_image_proj_mat: torch.Tensor  # world to image projection matrix, shape (4, 4)
    perspective_proj_mat: torch.Tensor  # perspective matrix, shape (4, 4)


@dataclass
class Sample:
    img: torch.Tensor  # image, shape (3, H, W)
    image_height: int  # image height
    image_width: int  # image width
    cam_idx: int  # camera id
    camera_center: torch.Tensor  # camera center, shape (3,)
    tan_half_fov_x: float  # tangent of half of the field of view in horizontal direction
    tan_half_fov_y: float  # tangent of half of the field of view in vertical direction
    world_to_view_proj_mat: torch.Tensor  # world to view projection matrix, shape (4, 4)
    world_to_image_proj_mat: torch.Tensor  # world to image projection matrix, shape (4, 4)
    screen_extent: float  # radius of the sphere that contains all camera centers


@dataclass
class RenderResults:
    rendered_img: torch.Tensor  # rendered image, shape (3, H, W)
    projected_means: torch.Tensor  # locations of Gaussians projected to 2D image plane, shape (N, 3)
    visibility_mask: torch.Tensor  # visibility mask of predicted Gaussian primitives, shape (L,)
    anchor_primitive_visible_mask: torch.Tensor  # visibility mask of anchor primitives derived by pre-filtering, shape (N,)
    pred_opacities: torch.Tensor  # predicted opacity before filtered by 0, shape (M, 1)
    coupled_primitive_mask: torch.Tensor  # mask of coupled primitives, shape (M, 1)
    scales: torch.Tensor  # scales of predicted Gaussian primitives, shape (M, 3)
    bpp: dict  # bits per pixel for each parameter to be compressed