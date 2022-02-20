#!/usr/bin/env python3

import numpy as np
import subprocess
import os
from pathlib import Path
import yaml

class DsoRun:
    def __init__(self, reverse):
        self.reverse_ = reverse

    def run(self):
        name = 'benchmark_results/' + self.method_ + '/' + self.dataset_.replace('/', '_')

        if self.reverse_:
            name += '_rev'

        try:
            os.mkdir('benchmark_results')
        except FileExistsError:
            pass

        try:
            os.mkdir('benchmark_results/' + self.method_)
        except FileExistsError:
            pass

        have_data = False
        retries = 0
        while not have_data and retries < 5:
            print(self.dataset_)
            subprocess.run(self.cmd_)

            try:
                # copy results with unique name
                os.rename('result.txt', name + '.txt')
                have_data = True
            except FileNotFoundError:
                print('')
                print('###########################')
                print('###########################')
                print('RUN FAILED, RETRYING.......')
                print('###########################')
                print('###########################')
                print('')
                retries += 1

class DsolRun(DsoRun):
    def __init__(self, dataset, dataset_calib, gt, reverse, dataset_id):
        super().__init__(reverse)
        self.method_ = 'dsol'
        self.dataset_ = dataset
        self.reverse_ = reverse
        self.dataset_id_ = dataset_id
        self.cmd_ = ['roslaunch', 'svcpp', 'dsol_data.launch', 'save:=' + str(Path.cwd()) + '/result.txt', 'data_dir:=' + dataset, 'data:=' + dataset_id]
        #if self.reverse_:
        #    self.cmd_.append('

class DsolBench:
    def __init__(self):
        self.datasets_ = yaml.safe_load(open('datasets/datasets_kitti.yml', 'r'))
    
    def run(self):
        for dataset in self.datasets_:
            for seq in dataset['sequences']:
                full_path = dataset['root'] + '/' + seq
                dr = DsolRun(full_path, dataset['calib'], dataset['gt'], False, dataset['dataset'])
                dr.run()
                #dr = DsolRun(full_path, dataset['calib'], dataset['gt'], True, dataset['dataset'])
                #dr.run()

if __name__ == '__main__':
    db = DsolBench()
    db.run()
