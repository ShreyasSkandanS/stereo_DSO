#!/usr/bin/env python

from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

basedir = Path('/tmp/vkitti')
scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
variations = [
    'clone', '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right',
    'fog', 'morning', 'overcast', 'rain', 'sunset'
]

save_path = Path('/tmp/vkitti_gt/')
save_path.mkdir(exist_ok=True)


def prep_extrinsics(file) -> np.ndarray:
    # i, cam, T00, T01, T02, T03, T11, ...
    data = np.loadtxt(file, skiprows=1)
    extrins = data[:, 2:].reshape(-1, 4, 4)
    # skip every other row
    extrins = extrins[::2]
    # fix rotation using scipy
    extrins[:, :3, :3] = Rotation.from_matrix(extrins[:, :3, :3]).as_matrix()
    # normalize extrin to the first camera
    # T_ci_c0  = T_ci_w @ T_c0_w^-1 = T_ci_w @ T_w_c0
    # extrins = extrins @ np.linalg.inv(extrins[0])
    return extrins


def inverse_transform(T: np.ndarray) -> np.ndarray:
    """Inverse transform [R, t]^-1 = [R', -R'@t]"""
    assert T.shape[-2:] == (4, 4), T.shape

    T_inv = np.zeros_like(T)
    R_t = np.swapaxes(T[..., :3, :3], -1, -2)
    T_inv[..., :3, :3] = R_t
    T_inv[..., :3, [3]] = -(R_t @ T[..., :3, [3]])
    T_inv[..., -1, -1] = 1.0

    return T_inv


def poses2tum(poses: np.ndarray) -> np.ndarray:
    # normalize poses
    poses = inverse_transform(poses[[0]]) @ poses
    quats = Rotation.from_matrix(poses[..., :3, :3]).as_quat()
    n = len(poses)

    tum_data = np.empty((n, 8))
    tum_data[:, 0] = np.arange(n)
    tum_data[:, 1:4] = poses[:, :3, 3]
    tum_data[:, 4:] = quats
    return tum_data


def write_tum(tum_data: np.ndarray, filename: str):
    np.savetxt(filename, tum_data, fmt="%d %.8f %.8f %.8f %.8f %.8f %.8f %.8f")


for scene in scenes:
    for variation in variations:
        print(f"Scene: {scene} variation: {variation}")
        extrin_file = basedir / scene / variation / 'extrinsic.txt'

        # get extrinsics then convert to poses
        extrins = prep_extrinsics(extrin_file)
        poses = inverse_transform(extrins)

        write_tum(poses2tum(poses), save_path / f"{scene}_{variation}.txt")
        write_tum(poses2tum(poses[::-1]),
                  save_path / f"{scene}_{variation}_rev.txt")
