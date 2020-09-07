#!/bin/bash
device=$1
#let start=$device*5
#let end=($device + 1)*5
let start=10
let end=15

for ((i=$start; i<$end; i++)) 
do
    echo "python train.py --dir data/models/ensemble/$i --data_dir /dev/shm/MICCAI_BraTS2020_TrainingData/ -a --epochs 50 --model MonoUNet --num_workers 8 --batch_size 5 --save_freq 1 --eval_freq 100 --device $1 --eclr"
    python train.py --dir data/models/ensemble/$i --data_dir /dev/shm/MICCAI_BraTS2020_TrainingData/ -a --epochs 50 --model MonoUNet --num_workers 8 --batch_size 5 --save_freq 1 --eval_freq 100 --device $1 --eclr
done
