import os
import torch
import torch.nn as nn
from models import *
from tqdm import tqdm
import numpy as np

import argparse
from torch.utils.data import DataLoader
from data_loader import BraTSAnnotationDataset
import os
import nibabel as nib

device = torch.device('cuda:1')

parser = argparse.ArgumentParser(description='Annotate BraTS data.')
parser.add_argement('--model_dir' type=str, required=True)
parser.add_argument('--data_dir', type=str,
        default='/shared/mrfil-data/cddunca2/brats2020/MICCAI_BraTS2020_ValidationData')
args = parser.parse_args()

for p, _, files in os.walk(f'{args.model_dir}/checkpoints/'):
    checkpoint_file = os.path.join(p, files[-1])
annotations_dir = f'{args.model_dir}/annotations'
os.makedirs(annotations_dir, exist_ok=True)

checkpoint = torch.load(checkpoint_file)
model = MonoUNet()
model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.to(device)

brats_data = BraTSAnnotationDataset(args.data_dir, 
        dims=[160, 192, 128])
dataloader = DataLoader(brats_data)

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
        
        label = torch.zeros((240, 240, 155))

        label[torch.where(wt > 0.5)] = 2
        label[torch.where(tc > 0.5)] = 1
        label[torch.where(et > 0.5)] = 4
        
        orig_header = nib.load(d['file'][0][0]).header
        aff = orig_header.get_qform()
        img = nib.Nifti1Image(label.numpy(), aff, header=orig_header)
        img.to_filename(os.path.join(annotations_dir, f'{d["patient"][0]}.nii.gz'))

