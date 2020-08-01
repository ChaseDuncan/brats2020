END=10
for ((i=0;i<=END;i++)); do
     echo "python thresh_sweep.py --seed 1234 -m $1 -o $2 -d /dev/shm/MICCAI_BraTS2020_TrainingData --thresh_idx $i --device 0"
     python thresh_sweep.py --seed 1234 -m $1 -o $2 -d /dev/shm/MICCAI_BraTS2020_TrainingData --thresh_idx $i --device 0
    done
