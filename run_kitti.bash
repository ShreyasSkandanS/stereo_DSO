#!/bin/bash
#usage: python xxx.py file_name

for((i=07;i<=07;i++))
  do
    {
     a=$((100+$i))
     seqnum=${a:1}
     echo "running seqence ${seqnum}"
     echo "./${usePriorPoses}/${seqnum}.txt"
    
    # run dso
     ./build/bin/dso_dataset \
 	  files=/home/shreyas/Work/kitti/dataset/sequences/${seqnum} \
 	  calib=/home/shreyas/stereo_DSO/calib/kitti/${seqnum}.txt \
	  groundtruth=/home/shreyas/Work/kitti/dataset/poses/${seqnum}.txt \
 	  preset=0 mode=1 \
	  quiet=1 nomt=1 nogui=1
    }&
  done


