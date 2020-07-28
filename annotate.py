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
parser.add_argument('-m', '--model_dir', type=str, required=True,
        help='Directory containing the model to use for annotation.')
parser.add_argument('-o', '--output_dir', type=str,
        default=None,
        help='Path to save annotations to (default: args.model_dir/annotations/')
parser.add_argument('-d', '--data_dir', type=str,
        default='/shared/mrfil-data/cddunca2/brats2020/MICCAI_BraTS2020_ValidationData',
        help='Path to directory of datasets to annotate (default: Brats 2020)')
parser.add_argument('-g', '--device', type=int, default=-1, metavar='N',
        help='Which device to use for annotation. (default: cpu)')
parser.add_argument('-t', '--thresh', type=float, default=0.5, metavar='p',
        help='threhold on probability for predicting true (default: 0.5)')
# finish this
#parser.add_argument('-c', '--checkpoint', type=int, default=None, metavar='N',
#        help='Which checkpoint to use if not most recent (default: most recent checkpoint in args.model_dir/checkpoints/)')

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

checkpoint = torch.load(checkpoint_file, map_location=device)
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

        label[torch.where(wt > args.thresh)] = 2
        label[torch.where(tc > args.thresh)] = 1
        label[torch.where(et > args.thresh)] = 4
        
        orig_header = nib.load(d['file'][0][0]).header
        aff = orig_header.get_qform()
        img = nib.Nifti1Image(label.numpy(), aff, header=orig_header)
        img.to_filename(os.path.join(annotations_dir, f'{d["patient"][0]}.nii.gz'))

