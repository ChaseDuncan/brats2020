import os
import torch
import torch.nn as nn
from models.models import *
from tqdm import tqdm
import numpy as np

import argparse
from torch.utils.data import DataLoader
from data_loader import BraTSAnnotationDataset
import os
import nibabel as nib


parser = argparse.ArgumentParser(description='Annotate BraTS data.')
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str,
        default=None)
parser.add_argument('--data_dir', type=str,
        default='/shared/mrfil-data/cddunca2/brats2020/MICCAI_BraTS2020_ValidationData')
parser.add_argument('--device', type=int, default=-1, metavar='N',
        help='Which device to use for annotation. (default: cpu)')
args = parser.parse_args()

if args.device >= 0:
    device = torch.device(f'cuda:{args.device}')
else:
    device = torch.device('cpu')

for p, _, files in os.walk(f'{args.model_dir}/checkpoints/'):
    checkpoint_file = os.path.join(p, files[-1])

if args.output_dir == None:
    annotations_dir = f'{args.model_dir}/annotations'
else:
    annotations_dir = f'{args.output_dir}'

os.makedirs(annotations_dir, exist_ok=True)

checkpoint = torch.load(checkpoint_file)
model = MonoUNet()
model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.to(device)

brats_data = BraTSAnnotationDataset(args.data_dir, 
        dims=[160, 192, 128])
dataloader = DataLoader(brats_data)
thresh = 0.25
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

        label[torch.where(wt > thresh)] = 2
        label[torch.where(tc > thresh)] = 1
        label[torch.where(et > thresh)] = 4
        
        orig_header = nib.load(d['file'][0][0]).header
        aff = orig_header.get_qform()
        img = nib.Nifti1Image(label.numpy(), aff, header=orig_header)
        img.to_filename(os.path.join(annotations_dir, f'{d["patient"][0]}.nii.gz'))

