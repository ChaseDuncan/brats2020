import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
from data_loader import BraTSValidation
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from models import MonoUNet
import models_min

from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from brats2018_dataloader import BraTS2018DataLoader3D, get_list_of_patients
device = torch.device('cuda:0')
model = MonoUNet()
#model = models_min.UNet()

model_name = 'monounet-bg'
cp = 300
#checkpoint = torch.load('data/models/checkpoints/checkpoint-10.pt')
#checkpoint = torch.load('data/models/monounet/checkpoints/checkpoint-5.pt')
#checkpoint = torch.load('data/models/min/checkpoints/checkpoint-300.pt')
#checkpoint = torch.load('data/models/dp-test/checkpoints/checkpoint-5.pt')
#checkpoint = torch.load('data/models/monounet/checkpoints/checkpoint-300.pt')
#checkpoint = torch.load('data/models/min-bg/checkpoints/checkpoint-300.pt')
#checkpoint = torch.load('data/models/monounet-baseline/checkpoints/checkpoint-300.pt')
#checkpoint = torch.load('data/models/monounet-bg/checkpoints/checkpoint-300.pt')
checkpoint = torch.load(f'data/models/{model_name}/checkpoints/checkpoint-{cp}.pt')
model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.to(device)
val = BraTSValidation('/dev/shm/brats2018-validation-preprocessed')
dataloader = DataLoader(val, batch_size=1, num_workers=0)
annotation_dir = f'annotations/{model_name}/2018/'
os.makedirs(annotation_dir, exist_ok=True)

thresh = 0.25
with torch.no_grad():
    model.eval()
    for data, metadata in tqdm(dataloader):
        orig_shape = np.array(data.size()[2:])
        # There are 4 spatial levels, i.e. 3 times in which the dimensionality of the 
        # input is halved. In order for the skip connections from the encoder to decoder
        # to work, the input must be divisible by 8. 
        new_shape = orig_shape + (8*np.ones(3)-(orig_shape % 8))

        # This is silly going back and forth from torch tensor to numpy array
        data = pad_nd_image(data, new_shape.astype('int'))

        output = model(torch.from_numpy(data).to(device)).squeeze()
        output = output.cpu().numpy()

        seg = np.zeros(output.shape)
        import pdb; pdb.set_trace()
        seg[np.where(output > thresh)] = 1
        seg = np.einsum('cijk->ijk', seg)
        seg[np.where(seg == 3)] = 4
        img = np.zeros(orig_shape)
        img[:, :, :] = seg[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
        output_file = os.path.join(annotation_dir, f'{metadata["patient_id"][0]}.nii.gz')
        BraTS2018DataLoader3D.save_segmentation_as_nifti(img, metadata, output_file)

