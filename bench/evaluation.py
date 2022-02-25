#!/usr/bin/env python

from scipy.spatial.transform import Rotation
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from evo.core import metrics, sync
from evo.tools import file_interface, plot
import matplotlib.pyplot as plt
import pickle


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


class EvalMethod:

    def __init__(self, base_dir: str, dataset: str, method: str):
        self.base_dir = base_dir
        self.dataset = dataset
        self.method = method
        self.method_path = Path(f'{self.base_dir}/{self.dataset}/{self.method}')
        self.gt_path = Path(f'{self.base_dir}/{self.dataset}/gt')
        self.gt_list = []
        self.method_list = []
        self.results_dict = {}

        self.save_dir = Path(f"{self.base_dir}/{self.dataset}/{self.method}/eval")
        self.save_dir.mkdir(exist_ok=True, parents=True)

        for gt_p in self.gt_path.iterdir():
            if gt_p.is_file():
                self.gt_list.append(gt_p)
        self.gt_list.sort()

        for m_p in self.method_path.iterdir():
            if m_p.is_file():
                self.method_list.append(m_p)
        self.method_list.sort()

        print(f"Identified {len(self.gt_list)} GT files and {len(self.method_list)} {self.method} files.")

    def evaluate_pair(self, method_path: Path, gt_path: Path) -> Results:
        gt_in = np.loadtxt(gt_path)
        method_in = np.loadtxt(method_path)

        traj_ref = file_interface.read_tum_trajectory_file(gt_path)
        traj_method = file_interface.read_tum_trajectory_file(method_path)

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

        res = Results(method=self.method, ape_rmse=ape_rmse_method, rpe_rmse=rpe_rmse_method,
                      ape_rmse_rot=ape_rmse_method_rot,
                      rpe_rmse_rot=rpe_rmse_method_rot, gt_len=gt_in.shape[0], method_len=method_in.shape[0])

        return res

    def evaluate_all(self):
        for method in self.method_list:
            gt = self.gt_path / f"{method.stem}.txt"
            assert (gt.is_file())
            res_pair = self.evaluate_pair(method, gt)
            self.results_dict[gt.stem] = res_pair

    def save_results(self):
        if self.results_dict:
            save_path = self.save_dir / "evaluate.pkl"
            outfile = open(save_path, 'wb')
            pickle.dump(self.results_dict, outfile)
            outfile.close()

    def load_results(self):
        load_path = self.save_dir / "evaluate.pkl"
        with open(load_path, 'rb') as f:
            p_data = pickle.load(f)
        if not self.results_dict:
            self.results_dict = p_data

    def visualize_results(self, thresh_complete, constant_error):
        RMSE_list = []
        for seq_k in self.results_dict.keys():
            print(f'Key: {seq_k}')
            sequence = self.results_dict[seq_k]
            print(f'Sequence: {sequence}\n')
            if sequence.method_len / sequence.gt_len >= thresh_complete:
                RMSE_list.append(sequence.ape_rmse)
            else:
                RMSE_list.append(constant_error)

        ape_rmse_viz_vector = np.tile(np.asarray(RMSE_list), (5, 1))
        fig = plt.figure()
        plt.imshow(ape_rmse_viz_vector, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.grid(None)
        plt.show()
