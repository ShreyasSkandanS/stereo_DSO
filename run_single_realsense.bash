#!/bin/bash
#usage: python xxx.py file_name

# run dso
./build/bin/dso_dataset \
    files=/home/chao/Documents/realsense \
    calib=/home/chao/Workspace/repo/versatran01/sdso/calib/realsense/calib.txt \
    groundtruth=/home/chao/Workspace/repo/versatran01/sdso/groundTruthPose/dummy.txt \
    preset=0 mode=1 \
    quiet=1 nomt=0
