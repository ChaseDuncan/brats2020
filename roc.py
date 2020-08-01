import os
import random
import argparse

import torch
import numpy as np
import nibabel as nib

from utils import cross_val

parser = argparse.ArgumentParser(description='Compute and graph ROC curves.')
parser.add_argument('-s', '--seg_dir', type=str, required=True,
        help='Path to directory containing segmentations from model.')
parser.add_argument('-d', '--data_dir', type=str,
        default='/dev/shm/MICCAI_BraTS2020_TrainingData',
        help='Path to ground truth (default: /dev/shm/MICCAI_BraTS2020_TrainingData)')
parser.add_argument('-o', '--output_dir', type=str, default='ROC.png',
        help='Path to save png (default: ./ROC.png')
parser.add_argument('--seed', type=int, default=1, metavar='S', 
    help='random seed (default: 1)')
parser.add_argument('-b', '--debug', action='store_true', 
        help='turn on debug mode which only uses first two files in each directory. (default: off)')
parser.add_argument('-g', '--device', type=int, default=-1, metavar='N', help='Which device to use (default: cpu)')
args = parser.parse_args()

if args.device >= 0:
    device = torch.device(f'cuda:{args.device}')
else:
    device = torch.device('cpu')

np.random.seed(args.seed)
random.seed(args.seed)
n = num_thresholds = 10
segs = []
_, (_, gts) = cross_val(args.data_dir) 

for (dirpath, dirnames, files) in os.walk(args.seg_dir):
  segs += [os.path.join(dirpath, file)\
          for file in files if '.nii.gz' in file ]

gts.sort()
debug_mod = len(gts)
gts*=num_thresholds
segs.sort()

# be sure we've got the correct ground truth datasets
gts_patients = sorted([gt.split('/')[-1] for gt in gts])
segs_patients = sorted([seg.split('/')[-1] for seg in segs])
for g, s in zip(gts_patients, segs_patients):
    assert g[:-11] == s[:-7], f'{g[:-11]}, {s[:-7]} do not match'

def roc_stats(seg_t, gt_t):
    st2 = torch.einsum('ijk, ijk->', seg_t, seg_t)
    tp = torch.einsum('ijk, ijk->', seg_t, gt_t)
    fp = st2 - tp
    fn = torch.einsum('ijk, ijk->', gt_t, gt_t) - tp
    tn = ((gt_t.size()[0]*gt_t.size()[1]*gt_t.size()[2]) - st2) - fp

    return tp / (tp + fn + 1e-32), fp / (fp + tn + 1e-32)

def conf_mats(seg, gt):
    ''' Compute true positives and false positives for the three clinical labels 
    enhancing tumor, whole tumor, and tumor core.
    '''
    o = torch.ones(seg.shape).cuda()
    z = torch.zeros(seg.shape).cuda()
    seg_et = torch.where(seg == 4, o, z)
    gt_et = torch.where(gt == 4, o, z)
    et = roc_stats(seg_et, gt_et)

    # seg_wt, gt_wt = seg[torch.where(seg > 0)], gt[torch.where(gt > 0)]
    seg_wt = torch.where(seg > 0, o, z)
    gt_wt = torch.where(gt > 0, o, z)
    wt = roc_stats(seg_wt, gt_wt)

    seg_tc = torch.where((seg == 4) | (seg == 1), 
            o, z)
    gt_tc = torch.where((gt== 4) | (gt == 1), 
            o, z)
    tc = roc_stats(seg_tc, gt_tc)

    return et, wt, tc
   
print('Computing confusion matrix.')
stats = []
for i, (seg, gt) in enumerate(zip(segs, gts)):
    if i % debug_mod != 0 and i % debug_mod != 1 and i % debug_mod != 2 and args.debug:
        continue
    print(f'processing... {i} / {len(segs)}      {seg}')
    # for some reason the 4s load in as 3.99999... so we must round up
    seg_d = torch.from_numpy(
            np.rint(
                nib.load(seg).get_fdata()
                )
            ).to(device)
    gt_d = torch.from_numpy(
            nib.load(gt).get_fdata()
            ).to(device)
    # list of triples with an index for each label type.
    # each element of the triple is a double 
    # (true positives, false positives).
    stats.append(conf_mats(seg_d, gt_d))

n = len(stats) // num_thresholds
by_thresh = [torch.tensor(stats[i:i + n])\
        for i in range(0, len(stats), n)]

thresholds = [(i+1) / 10 for i in range(10)]
for i, thresh in enumerate(by_thresh):
    N = len(thresh)    
    macro = torch.sum(thresh, axis=0) / N
    print(f'{thresholds[i]} {macro.view(-1).tolist()}')
    
