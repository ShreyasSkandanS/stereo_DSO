#!/usr/bin/env python3

import numpy as np
from evo.core import metrics, sync
from evo.tools import file_interface, plot
import matplotlib.pyplot as plt
import os

class GenMetrics:
    def __init__(self, data_dir):
        files = os.scandir(data_dir) 
        self.errors_ = []
        self.datasets_ = []
        gt_dir = '/tmp/vkitti_gt/'
        for file in files:
            print("Currently processing: {}".format(file.name))
            original_file = file.path

            # search for gt
            gt_filepath = ''
            for gt_file in os.scandir(gt_dir):
                #print("--- checking: {}".format(gt_file.name[:-4]))
                if gt_file.name[:-4].replace('-', '_') in original_file.lower().replace('-', '_') and '_rev' not in gt_file.name:
                    gt_filepath = gt_file.path
                    break
            if gt_filepath == '':
                print('Ground Truth ERROR: NO GROUND TRUTH FOUND FOR ' + str(original_file))
                break
            print("-- Found GT file: {}".format(gt_filepath))
            # create mirror file if necessary
            if file.name[-7:] == 'rev.txt':
                rev_gt_filepath = gt_filepath[:-4] + '_rev.txt'
                if not os.path.exists(rev_gt_filepath):
                    original_gt = open(gt_filepath, 'r').readlines()
                    flipped_gt = []
                    for enum, pose in enumerate(reversed(original_gt)):
                        pose_parsed = pose.split(' ')[1:]
                        flipped_gt.append(' '.join([str(enum), *pose_parsed]))
                    rev_gt_file = open(rev_gt_filepath, 'w')
                    rev_gt_file.writelines(flipped_gt)
                    rev_gt_file.close()
                gt_filepath = rev_gt_filepath

            traj_ref = file_interface.read_tum_trajectory_file(gt_filepath)
            traj_est = file_interface.read_tum_trajectory_file(original_file)

            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
            traj_est.align(traj_ref)

            ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
            ape_metric.process_data((traj_ref, traj_est))
            ape_rmse = ape_metric.get_statistic(metrics.StatisticsType.rmse)
            print(ape_rmse)
            self.datasets_.append(file.name)
            self.errors_.append(ape_rmse)

            #fig = plt.figure()
            #traj_by_label = {
            #    "estimate (aligned)": traj_est,
            #    "reference": traj_ref
            #}
            #plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
            #plt.show()

    def gen_cumulative_error_plot(self, ax, error_list, clr):
        error_less_count = np.sum(np.array(self.errors_)[:,None] < error_list, axis=0)
        ax.plot(error_list, error_less_count/len(self.errors_), clr)
        #plt.show()

if __name__ == '__main__':
    fig, ax = plt.subplots()
    error_list = np.arange(0, 200, 0.01)
    gm = GenMetrics('benchmark_results/dsol')
    gm.gen_cumulative_error_plot(ax, error_list, 'r')
    gm2 = GenMetrics('benchmark_results/stereo_dso')
    gm2.gen_cumulative_error_plot(ax, error_list, 'b')
    plt.show()
