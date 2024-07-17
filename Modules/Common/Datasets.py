import os
import struct
from collections import namedtuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .DataStructure import Sample, DatasetSample
from .Utils import CustomLogger

CameraModel = namedtuple(typename='CameraModel', field_names=['model_id', 'model_name', 'num_params'])
CAMERA_MODELS = {CameraModel(model_id=0, model_name='SIMPLE_PINHOLE', num_params=3), CameraModel(model_id=1, model_name='PINHOLE', num_params=4)}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])


class BaseDataset(Dataset):
    def __init__(self, root: str, image_folder: str, logger: CustomLogger, device: str,  eval_interval: int = 8, z_near: float = 0.01, z_far: float = 100) -> None:
        """
        Dataset for loading images and camera calibration matrices.
        :param root: path to the root folder of the dataset.
        :param image_folder: folder that contains images.
        :param logger: logger to record information.
        :param device: device to load data.
        :param eval_interval: interval of evaluated views, default is 8.
        :param z_near: distance between camera and near clipping plane, default is 0.01.
        :param z_far: distance between camera and far clipping plane, default is 100.
        """
        self.root, self.logger, self.device, self.z_near, self.z_far = root, logger, device, z_near, z_far
        self.logger.info('\nLoading data from {}...'.format(root))

        self.sfm_point_cloud_path = os.path.join(root, 'sparse', '0', 'points3D.bin')

        # load camera calibration information
        cam_mat_folder = os.path.join(root, 'sparse', '0')
        img_folder = os.path.join(root, image_folder)
        samples = sorted(self.load_samples(cam_mat_folder=cam_mat_folder, img_folder=img_folder), key=lambda x: x.img_name)

        self.train_samples = {}  # training samples
        self.test_samples = {}  # test samples

        # train / eval split
        self.train_samples = {idx: sample for idx, sample in enumerate(samples) if idx % eval_interval != 0}
        self.test_samples = {idx: sample for idx, sample in enumerate(samples) if idx % eval_interval == 0}

        # calculate the radius of the sphere that contains all training camera centers
        self.screen_extent = self.calculate_camera_sphere_radius()

        self.train_views_stack = list(self.train_samples.keys())  # stack to save views for training

        self.training = True  # flag to indicate if the dataset is in training mode

        logger.info(f'Loaded {len(self.train_samples)} training views and {len(self.test_samples)} evaluation views.')

    def __getitem__(self, idx: int) -> Sample:
        """
        Return image, intrinsic matrix and extrinsic matrix of the given index.
        :param idx: order of the data.
        :return sample: samples including image, camera_center, world_to_view_proj_mat, world_to_image_proj_mat, etc.
        """
        if not self.train_views_stack:  # re-create the training views stack
            self.logger.info('\nRe-creating training views stack...')
            self.train_views_stack = list(self.train_samples.keys())
        if self.training:  # randomly sample a view from the training views stack
            sample = self.train_samples[self.train_views_stack.pop(np.random.randint(0, len(self.train_views_stack)))]
        else:  # select a view from the test views
            sample = self.test_samples[list(self.test_samples.keys())[idx]]
        sample = Sample(img=sample.img, image_height=sample.img.shape[1], image_width=sample.img.shape[2],
                        tan_half_fov_x=sample.tan_half_fov_x, tan_half_fov_y=sample.tan_half_fov_y,
                        camera_center=sample.camera_center, cam_idx=sample.cam_idx, screen_extent=self.screen_extent,
                        world_to_view_proj_mat=sample.world_to_view_proj_mat,
                        world_to_image_proj_mat=sample.world_to_image_proj_mat)
        return sample

    def __len__(self) -> int:
        return len(self.train_samples) if self.training else len(self.test_samples)

    def eval(self) -> None:
        """
        Set the dataset to evaluation mode.
        """
        self.training = False

    def train(self) -> None:
        """
        Set the dataset to training mode.
        """
        self.training = True

    def load_samples(self, cam_mat_folder: str, img_folder: str) -> list[DatasetSample]:
        """
        Load camera calibration information and images.
        :param cam_mat_folder: path to the folder that contains camera calibration files.
        :param img_folder: path to the folder that contains images.
        """
        # load extrinsic data
        extrinsic_filepath = os.path.join(cam_mat_folder, 'images.bin')
        extrinsic_data = self.read_extrinsic_binary(extrinsic_filepath)

        img_names = {view_idx: extrinsic_data[view_idx]['img_name'] for view_idx in extrinsic_data}
        imgs = {view_idx: torch.tensor(np.array(Image.open(os.path.join(img_folder, img_name))).transpose(2, 0, 1) / 255., dtype=torch.float, device=self.device).clamp_(min=0., max=1.) for view_idx, img_name in img_names.items()}

        rotation_mats = {view_idx: extrinsic_data[view_idx]['rotation_mat'] for view_idx in extrinsic_data}
        translation_vectors = {view_idx: extrinsic_data[view_idx]['translation_vector'] for view_idx in extrinsic_data}

        # load intrinsic data
        intrinsic_filepath = os.path.join(cam_mat_folder, 'cameras.bin')
        intrinsic_data = self.read_intrinsic_binary(intrinsic_filepath)

        fov_x = {view_idx: intrinsic_data[extrinsic_data[view_idx]['camera_model_id']]['fov_x'] for view_idx in extrinsic_data}
        fov_y = {view_idx: intrinsic_data[extrinsic_data[view_idx]['camera_model_id']]['fov_y'] for view_idx in extrinsic_data}

        height = {view_idx: intrinsic_data[extrinsic_data[view_idx]['camera_model_id']]['height'] for view_idx in extrinsic_data}
        width = {view_idx: intrinsic_data[extrinsic_data[view_idx]['camera_model_id']]['width'] for view_idx in extrinsic_data}

        # calculate projection matrices and camera centers
        proj_matrices = {view_idx: self.calculate_proj_mats(
            rotation_mat=rotation_mats[view_idx], translation_vector=translation_vectors[view_idx], fov_x=fov_x[view_idx], fov_y=fov_y[view_idx]) for view_idx in extrinsic_data}

        cams_calib_info = [
            DatasetSample(
                cam_idx=view_idx, camera_center=proj_matrices[view_idx]['camera_center'],
                img_name=img_names[view_idx], img=imgs[view_idx], img_height=height[view_idx], img_width=width[view_idx],
                rotation_mat=rotation_mats[view_idx], translation_vec=translation_vectors[view_idx],
                fov_x=fov_x[view_idx], fov_y=fov_y[view_idx], tan_half_fov_x=np.tan(fov_x[view_idx] / 2), tan_half_fov_y=np.tan(fov_y[view_idx] / 2),
                world_to_view_proj_mat=proj_matrices[view_idx]['world_to_view_proj_mat'],
                world_to_image_proj_mat=proj_matrices[view_idx]['world_to_image_proj_mat'],
                perspective_proj_mat=proj_matrices[view_idx]['perspective_proj_mat'])
            for view_idx in extrinsic_data]
        return cams_calib_info

    def calculate_proj_mats(self, rotation_mat: np.ndarray, translation_vector: np.ndarray, fov_x: float, fov_y: float) -> dict:
        """
        Calculate projection matrices and camera centers.
        :param rotation_mat: rotation matrix with shape (3, 3).
        :param translation_vector: translation vector with shape (3,).
        :param fov_x: field of view in x direction.
        :param fov_y: field of view in y direction.
        :return world_to_view_proj_mat: world to view projection matrix with shape (4, 4).
        :return perspective_proj_mat: perspective projection matrix with shape (4, 4).
        :return world_to_image_proj_mat: world to image projection matrix with shape (4, 4).
        :return camera_center: camera center in world space with shape (3,).
        """
        # calculate world to view projection matrices
        world_to_view_proj_mat = self.calculate_world_to_view_proj_mat(rotation_mat=rotation_mat, translation_vector=translation_vector)
        world_to_view_proj_mat = torch.tensor(world_to_view_proj_mat.transpose(), dtype=torch.float32, device=self.device)

        # calculate camera center in world coordinate
        camera_center = torch.inverse(world_to_view_proj_mat)[3, :3]

        # calculate perspective projection matrices
        perspective_proj_mat = self.calculate_perspective_project_mat(fov_x=fov_x, fov_y=fov_y)
        perspective_proj_mat = torch.tensor(perspective_proj_mat, dtype=torch.float32, device=self.device).transpose(0, 1)

        # calculate world to image projection matrices
        world_to_image_proj_mat = torch.matmul(world_to_view_proj_mat, perspective_proj_mat)

        return {'world_to_view_proj_mat': world_to_view_proj_mat, 'perspective_proj_mat': perspective_proj_mat,
                'world_to_image_proj_mat': world_to_image_proj_mat, 'camera_center': camera_center}

    def calculate_camera_sphere_radius(self) -> float:
        """
        Calculate the radius of the sphere that contains all training camera centers, note that the center of the sphere is the average of all camera centers.
        """
        # calculate all camera centers with shape (3, num_views)
        world_to_view_proj_mats = [self.calculate_world_to_view_proj_mat(rotation_mat=sample.rotation_mat, translation_vector=sample.translation_vec).astype(np.float32) for sample in self.train_samples.values()]
        camera_centers = np.stack([np.linalg.inv(mat)[:3, -1] for mat in world_to_view_proj_mats], axis=1)
        # calculate the average of all camera centers
        average_camera_center = np.mean(camera_centers, axis=1, keepdims=True)  # shape (3, 1)
        # calculate the distance between all camera centers and the average of all camera centers
        distance = np.linalg.norm(camera_centers - average_camera_center, axis=0, keepdims=True)  # shape (1, num_views)
        # calculate the radius of the sphere that contains all camera centers
        radius = np.max(distance).item() * 1.1  # amplify the radius by 1.1
        return radius

    @staticmethod
    def calculate_world_to_view_proj_mat(rotation_mat: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
        """
        Calculate world to view projection matrix (actually extrinsic matrix).
        :param rotation_mat: rotation matrix with shape (3, 3).
        :param translation_vector: translation vector with shape (3,).
        :return camera_mat: camera matrix with shape (4, 4).
        """
        world_to_view_proj_mat = np.concatenate([rotation_mat.transpose(), translation_vector.reshape(3, 1)], axis=1)  # shape (3, 4)
        world_to_view_proj_mat = np.concatenate([world_to_view_proj_mat, np.array([[0, 0, 0, 1.]])], axis=0)  # shape (4, 4)
        return world_to_view_proj_mat.astype(np.float32)

    def calculate_perspective_project_mat(self,  fov_x: float, fov_y: float) -> np.ndarray:
        """
        Calculate perspective projection matrix
        :param fov_x: field of view in x direction.
        :param fov_y: field of view in y direction.
        :return: perspective projection matrix with shape (4, 4).
        """
        tan_half_fov_x, tan_half_fov_y = np.tan(fov_x / 2), np.tan(fov_y / 2)
        proj_mat = np.array([
            [1. / tan_half_fov_x, 0, 0, 0], [0, 1. / tan_half_fov_y, 0, 0],
            [0, 0, self.z_far / (self.z_far - self.z_near), -self.z_far * self.z_near / (self.z_far - self.z_near)], [0, 0, 1, 0]])
        return proj_mat

    def read_extrinsic_binary(self, extrinsic_filepath: str) -> dict:
        """
        Read extrinsic parameters from binary file.
        """
        extrinsic_info = {}
        with open(extrinsic_filepath, mode='rb') as fid:
            num_views = self.read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
            for _ in range(num_views):
                data = self.read_next_bytes(fid, num_bytes=64, format_char_sequence='idddddddi')
                view_id, rotation_quaternion_vector, translation_vector, camera_id = data[0], np.array(data[1:5]), np.array(data[5:8]), data[8]
                rotation_mat = self.quaternion_to_rotation_matrix(quaternion=rotation_quaternion_vector).transpose()

                img_name = ""
                current_char = self.read_next_bytes(fid, 1, 'c')[0]
                while current_char != b'\x00':  # look for the ASCII 0 entry
                    img_name += current_char.decode('utf-8')
                    current_char = self.read_next_bytes(fid, 1, 'c')[0]

                num_points_2d = self.read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
                _ = self.read_next_bytes(fid, num_bytes=24 * num_points_2d, format_char_sequence='ddq' * num_points_2d)

                extrinsic_info[view_id] = {'view_id': view_id, 'camera_model_id': camera_id, 'img_name': img_name,
                                           'rotation_mat': rotation_mat, 'translation_vector': translation_vector}
        return extrinsic_info

    def read_intrinsic_binary(self, intrinsic_path: str) -> dict:
        """
        Read intrinsic parameters from binary file.
        """
        intrinsic_info = {}
        with open(intrinsic_path, mode='rb') as fid:
            num_camera_models = self.read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
            for _ in range(num_camera_models):
                data = self.read_next_bytes(fid, num_bytes=24, format_char_sequence='iiQQ')
                _, camera_model_id, width, height = data[0], data[1], data[2], data[3]
                camera_model_name = CAMERA_MODEL_IDS[camera_model_id].model_name
                assert camera_model_name in CAMERA_MODEL_NAMES, f'Camera model {camera_model_name} is not supported.'
                num_camera_model_params = CAMERA_MODEL_IDS[camera_model_id].num_params
                params = self.read_next_bytes(fid, num_bytes=8 * num_camera_model_params, format_char_sequence="d" * num_camera_model_params)
                # calculate field of view
                fov_x = 2 * np.arctan(width / 2 / params[0])
                fov_y = 2 * np.arctan(height / 2 / (params[0] if camera_model_name == 'SIMPLE_PINHOLE' else params[1]))
                intrinsic_info[camera_model_id] = {'fov_x': fov_x, 'fov_y': fov_y, 'width': width, 'height': height}
        return intrinsic_info

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.
        :param quaternion: quaternion vector with shape (4,).
        :return rotation_mat: rotation matrix with shape (3, 3).
        """
        rotation_mat = np.array([
            [1 - 2 * quaternion[2] ** 2 - 2 * quaternion[3] ** 2, 2 * quaternion[1] * quaternion[2] - 2 * quaternion[0] * quaternion[3], 2 * quaternion[3] * quaternion[1] + 2 * quaternion[0] * quaternion[2]],
            [2 * quaternion[1] * quaternion[2] + 2 * quaternion[0] * quaternion[3], 1 - 2 * quaternion[1] ** 2 - 2 * quaternion[3] ** 2, 2 * quaternion[2] * quaternion[3] - 2 * quaternion[0] * quaternion[1]],
            [2 * quaternion[3] * quaternion[1] - 2 * quaternion[0] * quaternion[2], 2 * quaternion[2] * quaternion[3] + 2 * quaternion[0] * quaternion[1], 1 - 2 * quaternion[1] ** 2 - 2 * quaternion[2] ** 2]])
        return rotation_mat

    @staticmethod
    def read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"):
        """Read and unpack the next bytes from a binary file.
        :param fid: File object.
        :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
        :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
        :param endian_character: Any of {@, =, <, >, !}
        :return: Tuple of read and unpacked values.
        """
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)