#!/bin/bash
#usage: python xxx.py file_name

# run dso
./build/bin/dso_dataset \
    files=/home/shreyas/realsense/outdoor_slow \
    calib=/home/shreyas/stereo_DSO/calib/realsense/calib.txt \
    groundtruth=/home/shreyas/stereo_DSO/groundTruthPose/dummy.txt \
    mode=1 quiet=1 nomt=0 nolog=1 nogui=0 preset=5 

