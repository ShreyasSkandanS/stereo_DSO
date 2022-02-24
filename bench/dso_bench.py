#!/usr/bin/env python3

import numpy as np
import subprocess
import os
from pathlib import Path
import yaml
import pdb

class DsoRun:
    def __init__(self, reverse):
        self.reverse_ = reverse

    def run(self):
        pdb.set_trace()
        fname = 
        name = '/tmp/' + self.method_ + '/' + fname
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

class StereoDsoRun(DsoRun):
    def __init__(self, dataset, dataset_calib, gt, reverse=False):
        super().__init__(reverse)
        self.method_ = 'stereo_dso'
        self.dataset_ = dataset
        if "vkitti" in dataset.lower():
            dataset += '/frames/rgb'
        self.cmd_ = ['/tmp/dso_dataset', 'files=' + dataset, 'calib=' + dataset_calib, 'groundtruth=' + gt, 'mode=2', 'nogui=1', 'quiet=1', 'nolog=1', 'nomt=1']
        if self.reverse_:
            self.cmd_.append('reverse=1')

class DsoBench:
    def __init__(self):
        self.datasets_ = yaml.safe_load(open('datasets/datasets_vkitti.yml', 'r'))
    
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
