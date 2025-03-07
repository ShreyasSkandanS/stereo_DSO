#!/bin/bash
#usage: python xxx.py file_name

# run dso
./build/bin/dso_dataset \
    files=/home/chao/Workspace/dataset/kitti/dataset/sequences/06 \
    calib=/home/chao/Workspace/repo/versatran01/stereo-dso/calib/kitti/06.txt \
    groundtruth=/home/chao/Workspace/dataset/kitti/dataset/poses/06.txt \
    preset=0 mode=1 \
    quiet=1 nomt=1
