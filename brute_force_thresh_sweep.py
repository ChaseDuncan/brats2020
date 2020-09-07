import os
import torch
import torch.nn as nn
from models.models import *
from losses import dice_score
from tqdm import tqdm
import numpy as np
import random
import pickle

import argparse
from torch.utils.data import DataLoader
from data_loader import BraTSAnnotationDataset, BraTSTrainDataset
import os


parser = argparse.ArgumentParser(description='Annotate BraTS data.')
parser.add_argument('-m', '--model_dir', type=str, required=True,
        help='Directory containing the model to use for annotation.')
parser.add_argument('-o', '--output_dir', type=str,
        default=None,
        help='Path to save logs to (default: args.model_dir/thresh-logs/')
parser.add_argument('-d', '--data_dir', type=str,
        default='/shared/mrfil-data/cddunca2/brats2020/MICCAI_BraTS2020_TrainingData',help='Path to directory of datasets to annotate (default: Brats 2020)')
parser.add_argument('--seed', type=int, default=1, metavar='S', 
    help='random seed (default: 1)')
parser.add_argument('-g', '--device', type=int, default=-1, metavar='N',
        help='Which device to use for annotation. (default: cpu)')
parser.add_argument('--batch_size', type=int, default=1, metavar='N', 
    help='batch_size (default: 1)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', 
    help='number of workers to assign to dataloader (default: 4)')

args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)
    
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if args.device >= 0:
    device = torch.device(f'cuda:{args.device}')
else:
    device = torch.device('cpu')

for p, _, files in os.walk(f'{args.model_dir}/checkpoints/'):
    checkpoint_file = os.path.join(p, files[-1])

checkpoint = torch.load(checkpoint_file, map_location=device)
model = MonoUNet()
model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.to(device)

dims = [128, 128, 128]
val_data = BraTSTrainDataset(args.data_dir, dims=dims, augment_data=False)
dataloader = DataLoader(val_data, batch_size=args.batch_size, 
                        shuffle=True, num_workers=args.num_workers)

all_vals = []
with torch.no_grad():
    model.eval()
    for src, tgt in tqdm(dataloader):
        src = src.to(device, dtype=torch.float)
        tgt = tgt.to(device, dtype=torch.float)
        output = model(src)
        vals = []

        for thresh in np.linspace(0, 1, num=100, endpoint=False):
            preds = torch.zeros(output.size())
            preds = preds.to(device, dtype=torch.float)
            preds[torch.where(output > thresh)] = 1
            vals.append(dice_score(preds, tgt).cpu().numpy())

        all_vals.append(vals)
            
pkl_file = open(f'{args.model_dir}/thresh_logs.pkl', 'wb')
pickle.dump(all_vals, pkl_file)
             
