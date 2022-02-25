#!/usr/bin/env python

from scipy.spatial.transform import Rotation
import numpy as np

from pathlib import Path

basedir = '/home/shreyas/Work/kitti/dataset/poses'
seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

save_path = Path('./kitti_gt/')
save_path.mkdir(exist_ok=True)

for seq in seqs:
    file = open(basedir + '/' + seq + '.txt', 'r')
    outfile_path = Path.joinpath(save_path, 'kitti_gt_' + seq + '.txt')
    outfile = open(outfile_path, 'w')
    for ind, pose_str in enumerate(file.readlines()):
        raw = np.fromstring(pose_str, sep=' ').reshape(3, 4)
        R = Rotation.from_matrix(raw[:3, :3]).as_quat()
        outfile.write(f"{ind} {raw[0,3]} {raw[1,3]} {raw[2,3]} {R[0]} {R[1]} {R[2]} {R[3]}\n")
    outfile.close()


