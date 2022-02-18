#!/usr/bin/env python

from scipy.spatial.transform import Rotation
import numpy as np

basedir = '/home/shreyas/Work/vkitti'
scenes = ['Scene01','Scene02','Scene06','Scene18','Scene20']
seqs = ['15-deg-left','15-deg-right','30-deg-left','30-deg-right','clone','fog','morning','overcast','rain','sunset']

def prep_extrinsics(file) -> np.ndarray:
    # i, cam, T00, T01, T02, T03, T11, ...
    data = np.loadtxt(file, skiprows=1)
    extrins = data[:, 2:].reshape(-1, 4, 4)
    # fix rotation using scipy
    from scipy.spatial.transform import Rotation
    extrins[:, :3, :3] = Rotation.from_matrix(extrins[:, :3, :3]).as_matrix()
    # normalize extrin to the first camera
    # T_ci_c0  = T_ci_w @ T_c0_w^-1 = T_ci_w @ T_w_c0
    extrins = extrins @ np.linalg.inv(extrins[0])
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

for scene in scenes:
    for seq in seqs:
        file = open(basedir + '/' + scene + '/' + seq + '/extrinsic.txt', 'r')
        outfile_name = 'vkitti_groundtruth/' + scene.lower() + '_' + seq.replace('-', '_') + '.txt'
        outfile = open(outfile_name, 'w')

        extrins = prep_extrinsics(file)
        poses = inverse_transform(extrins)

        quatR = Rotation.from_matrix(poses[..., :3, :3]).as_quat()
        for ind in range(quatR.shape[0]):
            outfile.write(f"{ind} {poses[ind, 0, 3]} {poses[ind, 1, 3]} {poses[ind, 2, 3]} {quatR[ind, 0]} {quatR[ind, 1]} {quatR[ind, 2]} {quatR[ind, 3]}\n")
        #for ind, pose_str in enumerate(file.readlines()):
        #    raw = np.fromstring(pose_str, sep=' ').reshape(3, 4)
        #    R = Rotation.from_matrix(raw[:3, :3]).as_quat()
        #    outfile.write(f"{ind} {raw[0,3]} {raw[1,3]} {raw[2,3]} {R[0]} {R[1]} {R[2]} {R[3]}\n")
        outfile.close()


