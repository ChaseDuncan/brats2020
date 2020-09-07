# Calculate the percentage of labeled tumor lost when center
# cropping to 128x128x128.
import os

import nibabel as nib
import numpy as np

data_dir = '/dev/shm/MICCAI_BraTS2020_TrainingData'

filenames = []
for (dirpath, dirnames, files) in os.walk(data_dir):
    filenames += [os.path.join(dirpath, file) for file in files if '.nii.gz' in file ]
    segs = sorted([ f for f in filenames if "seg.nii.gz" in f ])

tot_ratio=[]
tot_segs = len(segs)
for i, seg in enumerate(segs):
    print( f'{i / tot_segs:.2f}', end='\r')
    img_mat = nib.load(seg).get_fdata()
    tot_labels = len(np.where(img_mat>0)[0])
    img_mat[56:-56, 56:-56, 14:-13] = 0
    mask_mat = np.zeros(img_mat.shape)
    mask_mat[np.where(img_mat>0)] = 1
    
    tot_ratio += [np.sum(mask_mat)/tot_labels]

median = np.median(tot_ratio)
mean = np.mean(tot_ratio)
mn = np.min(tot_ratio)
mx = np.max(tot_ratio)

print(f'mean: {mean}\tmedian: {median}\tmin: {mn}\tmax: {mx}')

