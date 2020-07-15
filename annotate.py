import os
import torch
import torch.nn as nn
from models import *
from tqdm import tqdm
import numpy as np

import argparse
from torch.utils.data import DataLoader
from data_loader import BraTSDataset
import os
import nibabel as nib

device = torch.device('cuda:1')

parser = argparse.ArgumentParser(description='Annotate BraTS data.')
parser.add_argument('--data')
args = parser.parse_args()

checkpoint_file='data/models/mono-oldpipe-2020/checkpoints/checkpoint-100.pt'
annotations_dir = 'annotations/mono-oldpipe-2020/'
os.makedirs(annotations_dir, exist_ok=True)

checkpoint = torch.load(checkpoint_file)
model = MonoUNet()
model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.to(device)

brats_data = BraTSDataset('/shared/mrfil-data/cddunca2/brats2020/MICCAI_BraTS2020_ValidationData', 
        dims=[160, 192, 128])
dataloader = DataLoader(brats_data, batch_size=1, num_workers=0)

with torch.no_grad():
    model.eval()
    dims=[160, 192, 128]
    for d in tqdm(dataloader):
        src = d['data'].to(device, dtype=torch.float)
        output = model(src)

        x_off = int((240 - dims[0]) / 2)
        y_off = int((240 - dims[1]) / 2)
        z_off = int((155 - dims[2]) / 2)

        m = nn.ConstantPad3d((z_off, z_off, y_off, y_off, x_off, x_off), 0)

        et = m(output[0, 0, :, :, :])
        wt = m(output[0, 1, :, :, :])
        tc = m(output[0, 2, :, :, :])
        
        <<< HEAD
        label = torch.zeros((240, 240, 155))

        label[torch.where(wt > 0.5)] = 2
        label[torch.where(tc > 0.5)] = 1
        label[torch.where(et > 0.5)] = 4
        
        orig_header = nib.load(d['file'][0][0]).header
        aff = orig_header.get_qform()
        img = nib.Nifti1Image(label.numpy(), aff, header=orig_header)
        img.to_filename(os.path.join(annotations_dir, f'{d["patient"][0]}.nii.gz'))

        # for batchgenerator preprocessed data
        #output_file = os.path.join(annotation_dir, f'{metadata["patient_id"][0]}.nii.gz')
        #BraTS2018DataLoader3D.save_segmentation_as_nifti(label, metadata, output_file)

