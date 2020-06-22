import os
import torch
import torch.nn as nn
from models import MonoUNet
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
from data_loader import BraTSDataset

import nibabel as nib
import models_min

device = torch.device('cuda:0')
#model = MonoUNet()
model = models_min.UNet()


#checkpoint = torch.load('data/models/checkpoints/checkpoint-10.pt')
#checkpoint = torch.load('data/models/monounet/checkpoints/checkpoint-5.pt')
checkpoint = torch.load('data/models/min/checkpoints/checkpoint-300.pt')
model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.cuda()

brats_data = \
    BraTSDataset('/dev/shm/brats2018validation/', dims=[128, 128, 128])
annotation_dir = 'annotations/min/'
os.makedirs(annotation_dir, exist_ok=True)
dataloader = DataLoader(brats_data, batch_size=1, num_workers=0)
dims=[128, 128, 128]
with torch.no_grad():
  model.eval()
  for src, tgt in tqdm(dataloader):
    ID = tgt[0].split("/")[5]
    src = src.to(device, dtype=torch.float)
    
    output = model(src)
    x_off = int((240 - dims[0]) / 4)*2
    y_off = int((240 - dims[1]) / 4)*2
    m = nn.ConstantPad3d((13, 14, x_off, x_off, y_off, y_off), 0)
    
    ncr_net = m(output[0, 0, :, :, :])
    ed = m(output[0, 1, :, :, :])
    et = m(output[0, 2, :, :, :])
    
    label = torch.zeros((240, 240, 155))
    label[torch.where(ncr_net > 0.5)] = 1
    label[torch.where(ed > 0.5)] = 2
    label[torch.where(et > 0.5)] = 4

    img = nib.Nifti1Image(label.numpy(), np.eye(4))
     
    img.to_filename(os.path.join(annotation_dir, ID))

