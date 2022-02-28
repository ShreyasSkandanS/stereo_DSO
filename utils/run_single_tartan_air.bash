#!/bin/bash
#usage: python xxx.py file_name

# run dso
./build/bin/dso_dataset \
    files=/media/chao/External/dataset/tartan_air/office/Easy/P000 \
    calib=/home/chao/Workspace/repo/versatran01/stereo-dso/calib/tartan_air.txt \
    groundtruth=/home/chao/Workspace/repo/versatran01/stereo-dso/groundTruthPose/dummy.txt \
    preset=0 mode=1 \
    quiet=1 nomt=1
