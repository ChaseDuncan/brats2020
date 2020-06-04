import os
import torch
import torch.nn as nn
from models import *
from tqdm import tqdm
import numpy as np

import argparse
from torch.utils.data import DataLoader
from data_loader import BraTSDataset
from utils import MRISegConfigParser
import os
import nibabel as nib

device = torch.device('cuda')

parser = argparse.ArgumentParser(description='Annotate BraTS data.')
parser.add_argument('--data')
args = parser.parse_args()

#checkpoint_file='/shared/mrfil-data/cddunca2/gliomaseg/baseline/checkpoints/checkpoint-300.pt'
#checkpoint_file = 'checkpoints/vaereg-fulltrain-vision/vaereg-fulltrain'
#checkpoint_file = 'checkpoints/vaereg-fulltrain/vaereg-fulltrain'
checkpoint_file='/shared/mrfil-data/cddunca2/brats2020/models/checkpoints/checkpoint-100.pt'
#annotations_dir = '/shared/mrfil-data/cddunca2/OSFData/NCM0014-segmentation/'
annotations_dir = '/shared/mrfil-data/cddunca2/annotations/brats2018validation'
os.makedirs(annotations_dir, exist_ok=True)

checkpoint = torch.load(checkpoint_file)
# Name of state dict in vision checkpoint
model = UNet(cfg)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.to(device)

brats_data = BraTSDataset('/dev/shm/brats2018validation/', dims=[160, 192, 128])
#brats_data = BraTSDataset('/shared/mrfil-data/cddunca2/OSFData/NCM0014/', 
#                      dims=[160, 192, 128])
dataloader = DataLoader(brats_data, batch_size=1, num_workers=0)

with torch.no_grad():
  model.eval()
  dims=[160, 192, 128]
  for src, tgt in tqdm(dataloader):
    ID = tgt[0].split("/")[-1]
    # This is ugly, loading in the image just to get its dimensions for uncropping
    src = src.to(device, dtype=torch.float)
    output = model(src)
    x_off = int((240 - dims[0]) / 2)
    y_off = int((240 - dims[1]) / 2)
    z_off = int((155 - dims[2]) / 2)
    m = nn.ConstantPad3d((z_off, z_off, y_off, y_off, x_off, x_off), 0)
    ncr_net = m(output[0, 0, :, :, :])
    ed = m(output[0, 1, :, :, :])
    et = m(output[0, 2, :, :, :])

    label = torch.zeros((240, 240, 155))

    label[torch.where(et > 0.5)] = 4
    label[torch.where(ed > 0.5)] = 2
    label[torch.where(ncr_net > 0.5)] = 1
    img = nib.Nifti1Image(label.numpy(), np.eye(4))
    #aff = np.array([[-1, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]])
    #img = nib.Nifti1Image(label.numpy(), aff)

    img.to_filename(os.path.join(annotations_dir, ID))

