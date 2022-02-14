#!/bin/bash
#usage: python xxx.py file_name

for((i=00;i<=00;i++))
  do
    {
     a=$((100+$i))
     seqnum=${a:1}
     echo "running seqence ${seqnum}"
     echo "./${usePriorPoses}/${seqnum}.txt"
    
    # run dso
     ./build/bin/dso_dataset \
 	  files=/home/shreyas/Work/kitti/dataset/sequences/${seqnum} \
 	  calib=/home/shreyas/Work/kitti/dataset/sequences/${seqnum}/calib.txt \
	  groundtruth=/home/shreyas/Work/kitti/dataset/poses/${seqnum}.txt \
 	  preset=0 mode=1 \
	  quiet=1 nomt=1
    }&
  done


