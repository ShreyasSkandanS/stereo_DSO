#!/usr/bin/env python

from scipy.spatial.transform import Rotation
import numpy as np

from pathlib import Path

basedir = Path('/home/shreyas/Work/tartan_air')
difficulty = ['Easy']

save_path = Path('/tmp/tartan_gt/')
save_path.mkdir(exist_ok=True)

def prep_poses(pose_file) -> np.ndarray:
    # tx ty tz qx qy qz qw
    data = np.loadtxt(pose_file)
    poses = np.zeros((data.shape[0], 4, 4))
    poses[:, :3, 3] = data[:, :3]
    poses[:, 3, 3] = 1
    # rotation from cam to ned
    R_ned_cam = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)

    from scipy.spatial.transform import Rotation
    # motion is defined in ned frame, but what we really need is motion of camera
    # T_w_c = T_w_ned @ R_ned_c
    poses[:, :3, :3] = Rotation.from_quat(data[:, 3:]).as_matrix() @ R_ned_cam
    # normalize pose to the first pose
    # T_c0_ci = T_w_c0^-1 @ T_w_ci = T_c0_w @ T_w_ci
    poses = np.linalg.inv(poses[0]) @ poses
    return poses

folders = list(basedir.glob('*'))
for scene in folders:
    for diff in difficulty:

        scene_path = Path.joinpath(basedir, scene.stem + '/' + diff)
        seq_list = list(scene_path.glob('*'))

        for seq in seq_list:
            print(f"Scene: {scene.stem} seq: {seq.stem}")
            pose_file_path = Path.joinpath(scene_path, seq.stem + '/pose_left.txt')
            file = open(pose_file_path, 'r')
            outfile_path = Path.joinpath(save_path, scene.stem.lower() + '_' + diff.lower() + '_' + seq.stem.lower() + '.txt')
            outfile = open(outfile_path, 'w')

            poses = prep_poses(file)

            quatR = Rotation.from_matrix(poses[..., :3, :3]).as_quat()
            for ind in range(quatR.shape[0]):
                outfile.write(f"{ind} {poses[ind, 0, 3]} {poses[ind, 1, 3]} {poses[ind, 2, 3]} {quatR[ind, 0]} {quatR[ind, 1]} {quatR[ind, 2]} {quatR[ind, 3]}\n")
            outfile.close()


