# Make directory on server and unzip data.

rm -rf /dev/shm/cddunca2/brats2020

mkdir -p /dev/shm/cddunca2/brats2020

cp /shared/rsaas/cddunca2/MICCAI_BraTS2020_TrainingData.zip /data/cddunca2/

unzip /data/cddunca2/MICCAI_BraTS2020_TrainingData.zip -d /dev/shm/cddunca2/brats2020/

