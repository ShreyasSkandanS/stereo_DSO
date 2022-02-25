#!/usr/bin/env python

from scipy.spatial.transform import Rotation
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from evo.core import metrics, sync
from evo.tools import file_interface, plot
import matplotlib.pyplot as plt
import pickle

dataset = 'vkitti'  # 'kitti', 'vkitti', 'tartan'
method = 'stereo_dso'  # 'dsol', 'stereo_dso'
base_dir = Path('/home/shreyas/stereo_DSO/bench/benchmark_results')
gt_dir = Path.joinpath(base_dir, dataset + '_gt')
method_dir = Path.joinpath(base_dir, method)
save_path = Path.joinpath(base_dir, method + '_' + dataset + '.pkl')

gt_reverse_flag = False


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

    def __init__(self, method: str, ape_rmse: float, rpe_rmse: float, ape_rmse_rot: float, rpe_rmse_rot: float,
                 gt_len: int, method_len: int):
        self.method = method
        self.ape_rmse = ape_rmse
        self.rpe_rmse = rpe_rmse
        self.ape_rmse_rot = ape_rmse_rot
        self.rpe_rmse_rot = rpe_rmse_rot
        self.gt_len = gt_len
        self.method_len = method_len


gt_list = list(gt_dir.glob('*'))
method_list = list(method_dir.glob('*_' + dataset + '_*'))

result_dict = {}

for gt_file in gt_list:
    assert (dataset in str(gt_file))
    sequence_id = gt_file.stem.split('_')[0]
    if dataset == 'vkitti':
        sequence_no = '_'.join(gt_file.stem.split('_')[1:])
        print(f'GT File: {gt_file.stem}, ID: {sequence_id}, #: {sequence_no}')
        dict_key = dataset + '_' + sequence_id + '_' + sequence_no
    elif dataset == 'tartan':
        sequence_no = gt_file.stem.split('_')[-1]
        print(f'GT File: {gt_file.stem}, ID: {sequence_id}, #: {sequence_no}')
        dict_key = dataset + '_' + sequence_id + '_' + sequence_no
    elif dataset == 'kitti':
        sequence_no = gt_file.stem.split('_')[-1]
        print(f'GT File: {gt_file.stem}, ID: {sequence_id}, #: {sequence_no}')
        dict_key = dataset + '_' + sequence_no
    else:
        print('ERROR. No such dataset.')
        break
    method_pair = ''

    # Open GT file
    gt_in = np.loadtxt(gt_file)

    if gt_file.stem.split('_')[-1] == 'rev':
        gt_reverse_flag = True
    else:
        gt_reverse_flag = False

    for method_file in method_list:
        if dataset == 'vkitti':
            method_ref_seq = method_file.stem.lower().replace('-', '_')
            method_ref_id = method_file.stem.lower()
        elif dataset == 'tartan':
            method_ref_seq = method_file.stem.lower()
            method_ref_id = method_file.stem
        elif dataset == 'kitti':
            method_ref_seq = method_file.stem.lower()
            method_ref_id = method_file.stem
        else:
            print('ERROR> No such dataset.')
            break

        if sequence_id in method_ref_id and sequence_no in method_ref_seq:
            if gt_reverse_flag:
                if method_file.stem.split('_')[-1] == 'rev':
                    method_pair = method_file
                else:
                    continue
            else:
                if method_file.stem.split('_')[-1] == 'rev':
                    continue
            method_pair = method_file
            break

    if method_pair == '':
        print('ERROR: Could not find Method reference file.\n')
        res = Results(method=method, ape_rmse=-1, rpe_rmse=-1, ape_rmse_rot=-1, rpe_rmse_rot=-1, gt_len=gt_in.shape[0],
                      method_len=-1)
        result_dict[dict_key] = res
        continue

    # Open DSOL file
    print(f'Method File: {method_file.stem}')
    method_in = np.loadtxt(method_pair)

    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    traj_method = file_interface.read_tum_trajectory_file(method_file)

    traj_ref, traj_method = sync.associate_trajectories(traj_ref, traj_method)
    traj_method.align(traj_ref)

    ape_metric_method = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric_method.process_data((traj_ref, traj_method))
    ape_rmse_method = ape_metric_method.get_statistic(metrics.StatisticsType.rmse)

    rpe_metric_method = metrics.RPE(metrics.PoseRelation.translation_part)
    rpe_metric_method.process_data((traj_ref, traj_method))
    rpe_rmse_method = rpe_metric_method.get_statistic(metrics.StatisticsType.rmse)

    ape_metric_method_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    ape_metric_method_rot.process_data((traj_ref, traj_method))
    ape_rmse_method_rot = ape_metric_method_rot.get_statistic(metrics.StatisticsType.rmse)

    rpe_metric_method_rot = metrics.RPE(metrics.PoseRelation.rotation_angle_deg)
    rpe_metric_method_rot.process_data((traj_ref, traj_method))
    rpe_rmse_method_rot = rpe_metric_method_rot.get_statistic(metrics.StatisticsType.rmse)

    res = Results(method=method, ape_rmse=ape_rmse_method, rpe_rmse=rpe_rmse_method, ape_rmse_rot=ape_rmse_method_rot,
                  rpe_rmse_rot=rpe_rmse_method_rot, gt_len=gt_in.shape[0], method_len=method_in.shape[0])
    result_dict[dict_key] = res

    print(f'Processed {dict_key} successfully.\n')

outfile = open(save_path, 'wb')
pickle.dump(result_dict, outfile)
outfile.close()
print(f'Method: {method} on dataset: {dataset} completed. Total data points: {len(result_dict)}')
