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

class StereoDsoRun(DsoRun):
    def __init__(self, dataset, dataset_calib, gt, reverse=False):
        super().__init__(reverse)
        self.method_ = 'stereo_dso'
        self.dataset_ = dataset
        if "vkitti" in dataset.lower():
            dataset += '/frames/rgb'
        self.cmd_ = ['/home/shreyas/stereo_DSO/build/bin/dso_dataset', 'files=' + dataset, 'calib=' + dataset_calib, 'groundtruth=' + gt, 'mode=1', 'nogui=1', 'quiet=1', 'nolog=1', 'nomt=0']
        if self.reverse_:
            self.cmd_.append('reverse=1')

class DsolRun(DsoRun):
    def __init__(self, dataset, dataset_calib, gt, reverse=False):
        super().__init__(reverse)
        self.method_ = 'dsol'
        self.dataset_ = dataset
        self.cmd_ = ['roslaunch', 'svcpp', 'dsol_data.launch', 'save:=' + str(Path.cwd()) + '/result.txt', 'base_dir:=' + dataset]

class DsoBench:
    def __init__(self):
        self.datasets_ = yaml.safe_load(open('datasets_vkitti.yml', 'r'))
    
    def run(self):
        for dataset in self.datasets_:
            for seq in dataset['sequences']:
                full_path = dataset['root'] + '/' + seq
                dr = StereoDsoRun(full_path, dataset['calib'], dataset['gt'])
                dr.run()
                dr = StereoDsoRun(full_path, dataset['calib'], dataset['gt'], True)
                dr.run()

if __name__ == '__main__':
    db = DsoBench()
    db.run()
