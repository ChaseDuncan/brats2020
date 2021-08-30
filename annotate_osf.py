import os
import torch
import torch.nn as nn
from models.models import *
from models.cascade_net import CascadeNet
from tqdm import tqdm
import numpy as np

import argparse
from torch.utils.data import DataLoader
from data_loader import BraTSAnnotationDataset, BraTSTrainDataset
import os
import nibabel as nib


parser = argparse.ArgumentParser(description='Annotate BraTS data.')
parser.add_argument('--dir', type=str, required=True,
        help='Data directory for model.')
parser.add_argument('--hier', type=str, default=None,
        help='Path to second model in hierarchy(default: None')
parser.add_argument('--output_dir', type=str, default=None,
        help='Path to save annotations to (default: args.model/annotations/{epoch}/')
parser.add_argument('--data_dir', type=str,
    default='/shared/mrfil-data/cddunca2/brats2020/MICCAI_BraTS2020_ValidationData',
        help='Path to directory of datasets to annotate (default: Brats 2020)')
parser.add_argument('-c', '--checkpoint', type=int, default=None, metavar='N',
        help='Specify a specific checkpoint. The default behavior is to use the\
                checkpoint with the largest epoch in its name.')
parser.add_argument('-g', '--device', type=int, default=-1, metavar='N',
        help='Which device to use for annotation. (default: cpu)')
parser.add_argument('-g2', '--device2', type=int, default=-1, metavar='N',
        help='Which device to use for second model in hierarchy. (default: cpu)')
parser.add_argument('--wt', type=float, default=0.5, metavar='p',
        help='threhold on probability for predicting true (default: 0.5)')
parser.add_argument('--et', type=float, default=0.5, metavar='p',
        help='threhold on probability for predicting true (default: 0.5)')
parser.add_argument('--tc', type=float, default=0.5, metavar='p',
        help='threhold on probability for predicting true (default: 0.5)')

parser.add_argument('-e', '--enhance_feat', action='store_true', 
    help='include t1ce/t1 in input and loss.')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                        help='model class (default: None)')
parser.add_argument('-F', '--full_patch', action='store_true', 
        help='use patch size 240x240x144 (default patch size: 128x128x128)')
parser.add_argument('-L', '--large_patch', action='store_true', 
        help='use patch size 160x192x128 (default patch size: 128x128x128)')
parser.add_argument('-X', '--xlarge_patch', action='store_true', 
        help='use patch size 192x192x128 (default patch size: 128x128x128)')

args = parser.parse_args()

dims=[128, 128, 128]
if args.full_patch:
    dims=[240, 240, 144]
if args.large_patch:
    dims=[160, 192, 128]
if args.xlarge_patch:
    dims=[192, 192, 128]

if args.device >= 0:
    device = torch.device(f'cuda:{args.device}')
else:
    device = torch.device('cpu')
brats_data = BraTSAnnotationDataset(args.data_dir, 
        dims=dims, enhance_feat=args.enhance_feat)
dataloader = DataLoader(brats_data)

for p, _, files in os.walk(f'{args.dir}/checkpoints/'):
    checkpoint_file = os.path.join(p, files[-1])
    if args.checkpoint is not None:
        for f in files:
            ep = ''.join([s for s in f if s.isdigit()])
            if args.checkpoint == int(ep):
                checkpoint_file = os.path.join(p, f)
                break

ep = ''.join([s for s in checkpoint_file if s.isdigit()])
if args.output_dir == None:
    annotations_dir = f'{args.dir}/annotations/{ep}/'
else:
    annotations_dir = f'{args.output_dir}'
seg_dir = f'{annotations_dir}/seg/'
unc_dir = f'{annotations_dir}/unc/'

os.makedirs(seg_dir, exist_ok=True)
os.makedirs(unc_dir, exist_ok=True)

if args.model.lower() == 'monounet':
    if args.enhance_feat:
        model = MonoUNet(input_channels=5)
    else:
        model = MonoUNet()
if args.model.lower() == 'vaereg':
    model = VAEReg()

checkpoint = torch.load(checkpoint_file, map_location=device)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.to(device)

#if args.model.lower() == 'average':
#    model = MonoUNet()
#    model = model_average(args.dir, model, device, sample_proportion=0.50, sample_rate=0.5)
#    model = model.to(device)
#    #brats_train_data = BraTSTrainDataset('/dev/shm/MICCAI_BraTS2020_TrainingData/', 
#    #        dims=dims, enhance_feat=args.enhance_feat)
#    #train_dataloader = DataLoader(brats_train_data)
#    #model = gn_update(train_dataloader, model, device)

#if args.model == 'CascadeNetLite':
#    model = CascadeNet(lite=True)
#    model.to(device)
#
#if args.hier:
#    model = HierarchicalNet(checkpoint_file, args.hier, device)
#    model.to(device)

#print('don\'t forget to change the thresholds back.')
# TODO: this shouldn't be hardcoded. the problem is that we do not store the original dimensions
# of the data in the data loader.
#src_dims = [240, 240, 156]
src_dims = [240, 240, 156]
with torch.no_grad():
    model.eval()
    for d in tqdm(dataloader):
        src = d['data'].to(device, dtype=torch.float)
        output, _ = model(src)
        if args.model == 'CascadeNetLite':
            output = output['biline']
        if isinstance(model, VAEReg):
            output = output['seg_map']
        x_off = (182 - dims[0]) // 2
        y_off = (218 - dims[1]) // 2
        z_off = (182 - dims[2]) // 2
        m = nn.ConstantPad3d((z_off, z_off-1, y_off, y_off, x_off, x_off), 0)
        et = m(output[0, 0, :, :, :])
        wt = m(output[0, 1, :, :, :])
        tc = m(output[0, 2, :, :, :])
        label = torch.zeros((182, 218, 182), dtype=torch.short)
        label[torch.where(wt > args.wt)] = 2
        label[torch.where(tc > args.tc)] = 1
        label[torch.where(et > args.et)] = 4

        label = label.long()

        unc_enhance = (100*(torch.ones(et.size()).to(device) - et)).type(torch.CharTensor)
        unc_whole = (100*(torch.ones(wt.size()).to(device) - wt)).type(torch.CharTensor)
        unc_core = (100*(torch.ones(tc.size()).to(device) - tc)).type(torch.CharTensor)

        '''
        The participants are called to upload 4 nifti (.nii.gz) volumes (3 
        uncertainty maps and 1 multi-class segmentation volume from Task 1) 
        onto CBICA's Image Processing Portal format. For example, for each ID in the 
        dataset, participants are expected to upload following 4 volumes:

        1. {ID}.nii.gz (multi-class label map)
        2. {ID}_unc_whole.nii.gz (Uncertainty map associated with whole tumor)
        3. {ID}_unc_core.nii.gz (Uncertainty map associated with tumor core)
        4. {ID}_unc_enhance.nii.gz (Uncertainty map associated with enhancing tumor)
        '''
        orig_header = nib.load(d['file'][0][0]).header
        aff = orig_header.get_qform()
        patient = d["patient"][0]

        label_map = nib.Nifti1Image(label.numpy(), aff, header=orig_header)
        print(orig_header)
        sys.exit()
        label_map.to_filename(os.path.join(seg_dir, f'{patient}.nii.gz'))
        
        # not sure about affine transform and header for this file
        unc_enhance = nib.Nifti1Image(unc_enhance.numpy(), aff, header=orig_header)
        unc_enhance.to_filename(os.path.join(unc_dir, f'{patient}_unc_enhance.nii.gz'))
          
        # not sure about affine transform and header for this file
        unc_whole = nib.Nifti1Image(unc_whole.numpy(), aff, header=orig_header)
        unc_whole.to_filename(os.path.join(unc_dir, f'{patient}_unc_whole.nii.gz'))

        # not sure about affine transform and header for this file
        unc_core = nib.Nifti1Image(unc_core.numpy(), aff, header=orig_header)
        unc_core.to_filename(os.path.join(unc_dir, f'{patient}_unc_core.nii.gz'))

