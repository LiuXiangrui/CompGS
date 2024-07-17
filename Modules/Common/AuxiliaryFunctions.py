import struct
import os
from tempfile import TemporaryDirectory

import numpy as np
import torch
from knn_dist import distance
from plyfile import PlyElement, PlyData


def voxelize_sample(points: np.ndarray, voxel_size: float = 0.01) -> np.ndarray:
    """
    Voxelize the input points.
    :param points: input points, shape (N, 3).
    :param voxel_size: voxel size.
    """
    np.random.shuffle(points)
    voxels = np.unique(np.round(points / voxel_size), axis=0) * voxel_size
    return voxels


def adaptive_voxel_size(points: np.ndarray) -> float:
    """
    Compute the adaptive voxel size for the input points.
    :param points: input points, shape (N, 3).
    """
    points = torch.tensor(points, dtype=torch.float32, device='cuda')
    dist = distance(points)
    median_dist = torch.median(dist)
    torch.cuda.empty_cache()
    return median_dist.item()


def load_point_clouds(point_cloud_path: str) -> tuple:
    """
    Load point clouds from binary file.
    :param point_cloud_path: path to the binary file.
    :return: points and colors of the point clouds.
    """
    assert point_cloud_path.endswith('.bin'), f"Point cloud file {point_cloud_path} must be a binary file."
    points, colors = [], []
    with open(point_cloud_path, 'rb') as f:
        num_points = read_bytes(f, num_bytes=8, fmt="Q")[0]  # number of points
        for idx in range(num_points):
            point_info = read_bytes(f, num_bytes=43, fmt="QdddBBBd")
            point = np.array(point_info[1:4])  # spatial location of the point (x, y, z)
            color = np.array(point_info[4:7])  # color of the point (r, g, b)
            _ = np.array(point_info[7])  # error of the point, just ignore it
            track_length = read_bytes(f, num_bytes=8, fmt="Q")[0]  # track length, just ignore it
            _ = read_bytes(f, num_bytes=8 * track_length, fmt="ii" * track_length)  # track elements, just ignore it
            points.append(point)
            colors.append(color)
    points = np.stack(points, axis=0)  # shape (N, 3)
    colors = np.stack(colors, axis=0)  # shape (N, 3)
    return points, colors


def save_point_clouds(ply_filepath: str, points: np.ndarray, colors: np.ndarray) -> None:
    """
    Save point clouds to ply file.
    :param ply_filepath: path to the ply file.
    :param points: points of the point clouds with shape (N, 3).
    :param colors: colors of the point clouds with shape (N, 3).
    """
    assert ply_filepath.endswith('.ply'), f"Point cloud file {ply_filepath} must be a ply file."
    data_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(points)
    elements = np.empty(points.shape[0], dtype=data_type)
    attributes = np.concatenate([points, normals, colors], axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_filepath)


def read_bytes(f, num_bytes: int, fmt: str, endian_character: str = "<"):
    """Read and unpack bytes from a binary file.
    :param f: file object.
    :param num_bytes: bytes to be read.
    :param fmt: format of the bytes to be read
    :param endian_character: endian character (default: little endian)
    :return: Tuple of read and unpacked values in the given format.
    """
    data = f.read(num_bytes)
    return struct.unpack(endian_character + fmt, data)


def gpcc_encode(encoder_path: str, ply_path: str, bin_path: str) -> None:
    """
    Compress geometry point cloud by GPCC codec.
    """
    enc_cmd = (f'{encoder_path} '
               f'--mode=0 --trisoupNodeSizeLog2=0 --mergeDuplicatedPoints=0 --neighbourAvailBoundaryLog2=8 '
               f'--intra_pred_max_node_size_log2=6 --positionQuantizationScale=1 --inferredDirectCodingMode=1 '
               f'--maxNumQtBtBeforeOt=4 --minQtbtSizeLog2=0 --planarEnabled=0 --planarModeIdcmUse=0 '
               f'--uncompressedDataPath={ply_path} --compressedStreamPath={bin_path} ')
    enc_cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    exit_code = os.system(enc_cmd)
    assert exit_code == 0, f'GPCC encoder failed with exit code {exit_code}.'


def gpcc_decode(decoder_path: str, bin_path: str, recon_path: str) -> None:
    """
    Decompress geometry point cloud by GPCC codec.
    """
    dec_cmd = (f'{decoder_path} '
               f'--mode=1 --outputBinaryPly=1 '
               f'--compressedStreamPath={bin_path} --reconstructedDataPath={recon_path} ')
    dec_cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    exit_code = os.system(dec_cmd)
    assert exit_code == 0, f'GPCC decoder failed with exit code {exit_code}.'


def write_ply_geo_ascii(geo_data: np.ndarray, ply_path: str) -> None:
    """
    Write geometry point cloud to a .ply file in ASCII format.
    """
    assert ply_path.endswith('.ply'), 'Destination path must be a .ply file.'
    assert geo_data.ndim == 2 and geo_data.shape[1] == 3, 'Input data must be a 3D point cloud.'
    geo_data = geo_data.astype(int)
    with open(ply_path, 'w') as f:
        # write header
        f.writelines(['ply\n', 'format ascii 1.0\n', f'element vertex {geo_data.shape[0]}\n',
                      'property float x\n', 'property float y\n', 'property float z\n', 'end_header\n'])
        # write data
        for point in geo_data:
            f.write(f'{point[0]} {point[1]} {point[2]}\n')


def read_ply_geo_bin(ply_path: str) -> np.ndarray:
    """
    Read geometry point cloud from a .ply file in binary format.
    """
    assert ply_path.endswith('.ply'), 'Source path must be a .ply file.'

    ply_data = PlyData.read(ply_path).elements[0]
    means = np.stack([ply_data.data[name] for name in ['x', 'y', 'z']], axis=1)  # shape (N, 3)
    return means


def calculate_morton_order(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate Morton order of the input points.
    """
    assert len(x.shape) == 2 and x.shape[1] == 3, f'Input data must be a 3D point cloud, but got {x.shape}.'
    device = x.device
    x = x.cpu().numpy()
    indices_sorted = np.argsort(x @ np.power(x.max() + 1, np.arange(x.shape[1])), axis=0)
    indices_sorted = torch.tensor(indices_sorted, dtype=torch.long, device=device)
    return indices_sorted


def compress_gpcc(x: torch.Tensor, gpcc_codec_path: str) -> bytes:
    """
    Compress geometry point cloud by GPCC codec.
    """
    assert len(x.shape) == 2 and x.shape[1] == 3, f'Input data must be a 3D point cloud, but got {x.shape}.'

    with TemporaryDirectory() as temp_dir:
        ply_path = os.path.join(temp_dir, 'point_cloud.ply')
        bin_path = os.path.join(temp_dir, 'point_cloud.bin')
        write_ply_geo_ascii(x.cpu().numpy(), ply_path=ply_path)
        gpcc_encode(encoder_path=gpcc_codec_path, ply_path=ply_path, bin_path=bin_path)
        with open(bin_path, 'rb') as f:
            strings = f.read()
    return strings


def decompress_gpcc(strings: bytes, gpcc_codec_path: str) -> torch.Tensor:
    """
    Decompress geometry point cloud by GPCC codec.
    """
    with TemporaryDirectory() as temp_dir:
        ply_path = os.path.join(temp_dir, 'point_cloud.ply')
        bin_path = os.path.join(temp_dir, 'point_cloud.bin')
        with open(bin_path, 'wb') as f:
            f.write(strings)
        gpcc_decode(decoder_path=gpcc_codec_path, bin_path=bin_path, recon_path=ply_path)
        x = read_ply_geo_bin(ply_path=ply_path)
    return torch.tensor(x, dtype=torch.float32)