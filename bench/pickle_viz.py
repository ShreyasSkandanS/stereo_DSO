#!/usr/bin/env python

from scipy.spatial.transform import Rotation
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from evo.core import metrics, sync
from evo.tools import file_interface, plot
import matplotlib.pyplot as plt
import pickle

dataset = 'vkitti'   # 'kitti', 'vkitti', 'tartan'
method = 'stereo_dso'      # 'dsol', 'stereo_dso'
base_dir = Path('/home/shreyas/stereo_DSO/bench/benchmark_results')
gt_dir = Path.joinpath(base_dir, dataset + '_gt')
method_dir = Path.joinpath(base_dir, method)
pickle_path = Path.joinpath(base_dir, method + '_' + dataset + '.pkl')
thresh = 0.8
LARGE_ERROR_CONSTANT = 150

@dataclass
class Results:
    """Class for book keeping results"""
    method: str
    ape_rmse: float
    rpe_rmse: float
    ape_rmse_rot: float
    rpe_rmse_rot: float
    gt_len: int
    method_len: int

    def __init__(self, method: str, ape_rmse: float, rpe_rmse: float, ape_rmse_rot: float, rpe_rmse_rot: float, gt_len: int, method_len: int):
        self.method = method
        self.ape_rmse = ape_rmse
        self.rpe_rmse = rpe_rmse
        self.ape_rmse_rot = ape_rmse_rot
        self.rpe_rmse_rot = rpe_rmse_rot
        self.gt_len = gt_len
        self.method_len = method_len

with open(pickle_path, 'rb') as f:
    p_data = pickle.load(f)

#print(f'Pickle data: {p_data}')

RMSE_list = []

for seq_k in p_data.keys():
    if seq_k.endswith('_rev'):
        continue
    print(f'Key: {seq_k}')
    sequence = p_data[seq_k]
    print(f'Sequence: {sequence}')
    if sequence.method_len / sequence.gt_len >= thresh:
        RMSE_list.append(sequence.ape_rmse)
    else:
        RMSE_list.append(LARGE_ERROR_CONSTANT)

#ape_rmse_viz_vector = np.tile(np.asarray(RMSE_list), (5,1))
ape_rmse_viz_vector = np.asarray(RMSE_list).reshape(5,10)

fig = plt.figure()
plt.imshow(ape_rmse_viz_vector, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.grid(None)
plt.show()