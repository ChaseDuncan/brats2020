import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import argparse

from torch.utils.data import DataLoader
from data_loader import BraTSValidation
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from models import MonoUNet
from lean_net import LeaNet
from dropout_lean_net import DropoutLeaNet

from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from brats2018_dataloader import BraTS2018DataLoader3D, get_list_of_patients
from utils import get_free_gpu

parser = argparse.ArgumentParser(description='Annotate BraTS data from a given year.')
device = torch.device(f'cuda:{get_free_gpu()}')

parser.add_argument('--model', type=str, required=True, metavar='CLASS',
    help='Which model class to use.')

parser.add_argument('--model_name', type=str, required=True, metavar='NAME',
    help='Name of model to use for annotation.')

parser.add_argument('--cp', type=int, metavar='N',
    help='Checkpoint to use for annotation. Optional.\
            Default behavior is to use the checkpoint with highest epoch\
            in the checkpoints directory for model NAME.')

parser.add_argument('--year', type=int, required=True, metavar='N',
    help='Year of BraTS data to label. Used for creating naming directory to store\
            output annotations.')

parser.add_argument('--data_dir', type=str, help='Directory of validation data.')

parser.add_argument('--thresh', type=float, default='0.5',
        help='The threshold for determining if a voxel should be labeled. (default: 0.5)')

args = parser.parse_args()

if args.model == 'MonoUNet':
    model = MonoUNet()
if args.model == 'LeaNet':
    model = LeaNet()
if args.model == 'LeaNetWithDropout':
    model = DropoutLeaNet()
if not args.cp:
    args.cp = max([int(cpt[:-3].split('-')[-1]) 
        for cpt in os.listdir(f'data/models/{args.model_name}/checkpoints/')])
checkpoint =\
        torch.load(f'data/models/{args.model_name}/checkpoints/checkpoint-{args.cp}.pt')

model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.to(device)

#val = BraTSValidation(f'/dev/shm/brats{args.year}-validation-preprocessed')
val = BraTSValidation(args.data_dir)
print(len(val))
dataloader = DataLoader(val, batch_size=1, num_workers=0)
annotation_dir = f'annotations/{args.model_name}/{args.year}/'
os.makedirs(annotation_dir, exist_ok=True)

with torch.no_grad():
    model.eval()
    for data, metadata in tqdm(dataloader):
        orig_shape = np.array(data.size()[2:])
        # There are 4 spatial levels, i.e. 3 times in which the dimensionality of the 
        # input is halved. In order for the skip connections from the encoder to decoder
        # to work, the input must be divisible by 8. 
        diff=(8*np.ones(3)-(orig_shape % 8)).astype('int')
        new_shape = orig_shape + diff

        # This is silly going back and forth from torch tensor to numpy array
        # I don't really even need this general function here. I think I can
        # simplify this using pytorch lib.
        data = pad_nd_image(data, new_shape.astype('int'))

        output = model(torch.from_numpy(data).to(device)).squeeze()
        output = output.cpu().numpy()
        output = output[:, :orig_shape[0], :orig_shape[1], :orig_shape[2]]
        seg = np.zeros(orig_shape)
        ncr_net = output[0, :, :, :]
        ed = output[1, :, :, :]
        et = output[2, :, :, :]
    
        label = np.zeros((orig_shape[0], orig_shape[1], orig_shape[2]))
        label[np.where(ncr_net > args.thresh)] = 1
        label[np.where(ed > args.thresh)] = 2
        label[np.where(et > args.thresh)] = 4
        
        output_file = os.path.join(annotation_dir, f'{metadata["patient_id"][0]}.nii.gz')
        BraTS2018DataLoader3D.save_segmentation_as_nifti(label, metadata, output_file)

