import os
import sys
import time
import tabulate

from torchcontrib.optim import SWA

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle
import argparse
import random
from models.cascade_net import CascadeNet
from torch.utils.data import DataLoader
from scheduler import PolynomialLR
import losses

from utils import *
from models.models import *
from data_loader import BraTSTrainDataset, BraTSSelfTrainDataset

#from apex import amp
from apex_dummy import amp

parser = argparse.ArgumentParser(description='Train glioma segmentation model.')
lr_add_cnst = 1e-6
# In this directory is stored the script used to start the training,
# the most recent and best checkpoints, and a directory of logs.
parser.add_argument('--dir', type=str, required=True, metavar='PATH',
    help='The directory to write all output to.')

parser.add_argument('--data_dir', type=str, default='/dev/shm/MICCAI_BraTS2020_TrainingData', metavar='PATH TO DATA',
    help='Path to where the data is located.')

parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                        help='model class (default: None)')

parser.add_argument('--coarse_wt_only', action='store_true', 
    help='only use whole tumor for loss function for coarse layer of CascadeNet.\
            only meaningful if args.model is CascadeNet (default: off)')

parser.add_argument('--device', type=int, required=True, metavar='N',
    help='Which device to use for training.')

parser.add_argument('--upsampling', type=str, default='bilinear', 
    choices=['bilinear', 'deconv'], 
    help='upsampling algorithm to use in decoder (default: bilinear)')

parser.add_argument('--loss', type=str, default='avgdice', 
    choices=['avgdice', 'vae', 'bce', 'dicebce'], 
    help='which loss to use during training (default: avgdice)')

parser.add_argument('-a', '--augment_data', action='store_true', 
    help='augment training data with mirroring, shifts, and scaling (default: off)')

parser.add_argument('--throw_no_et_sets', action='store_true', 
    help='throw out datasets that do not have ET labels (default: off)')

parser.add_argument('--mixed_precision', action='store_true', 
    help='mixed precision flag (default: off)')

parser.add_argument('--cascade_train', action='store_true', 
    help='train cascade model, append wt annotation to input (default: off)')

parser.add_argument('--instance_norm', action='store_true', 
    help='use instance normalization instead of group normalization (default: off)')

parser.add_argument('--clr', action='store_true', 
    help='use cyclical learning rate(default: off)')

parser.add_argument('--eclr', action='store_true', 
    help='step clr per epoch(default: off)')

parser.add_argument('--seedtest', action='store_true', 
    help='test for random seeds (default: off)')

parser.add_argument('-b', '--debug', action='store_true', 
    help='use debug mode which only uses a couple examples for training and testing (default: off)')

parser.add_argument('-L', '--large_patch', action='store_true', 
        help='use patch size 160x192x128 (default patch size: 128x128x128)')
parser.add_argument('-X', '--xlarge_patch', action='store_true', 
        help='use patch size 192x192x128 (default patch size: 128x128x128)')

parser.add_argument('-F', '--full_patch', action='store_true', 
        help='use patch size 240x240x144 (default patch size: 128x128x128)')

parser.add_argument('--cross_val', action='store_true', 
    help='use train/val split of full dataset (default: off)')

parser.add_argument('--swa', type=int, default=0, metavar='n', 
    help='use stochastic weight averaging during training.\
            n=0 for no swa, n>0 use swa starting at epoch n. (default: off)')

parser.add_argument('-e', '--enhance_feat', action='store_true', 
    help='include t1ce/t1 in input and loss.')

parser.add_argument('--selftrain', action='store_true',
        help='use Decathlon dataset for self-training.')

parser.add_argument('--selftrain_n', type=int, default=50, metavar='n',
        help='number of unsupervised examples to use for self train. (default: 50)')

rand_seed = random.randint(0, 2**32-1)
parser.add_argument('--seed', type=int, default=rand_seed, metavar='S', 
    help='random seed (default: random.randint(0, 2**32 - 1))')

parser.add_argument('--wd', type=float, default=1e-5, 
    help='weight decay (default: 1e-5)')

parser.add_argument('--resume', type=str, default=None, metavar='PATH',
                        help='checkpoint to resume training from (default: None)')

parser.add_argument('--pretrain', type=str, default=None, metavar='PATH',
                        help='pretrained model to start training from (default: None)')

parser.add_argument('--epochs', type=int, default=300, metavar='N', 
    help='number of epochs to train (default: 300)')

parser.add_argument('--num_workers', type=int, default=4, metavar='N', 
    help='number of workers to assign to dataloader (default: 4)')

parser.add_argument('--batch_size', type=int, default=1, metavar='N', 
    help='batch_size (default: 1)')

parser.add_argument('--save_freq', type=int, default=25, metavar='N', 
    help='save frequency (default: 25)')

parser.add_argument('--eval_freq', type=int, default=5, metavar='N', 
    help='evaluation frequency (default: 25)')

parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', 
    help='initial learning rate (default: 1e-4)')

parser.add_argument('--lr_add_cnst', type=float, default=1e-6, metavar='LR', 
    help='constant to add to learning rate each epoch (default: 1e-6)')

parser.add_argument('-m', '--message', action='store', dest='msg', nargs='*', type=str,
        help='a message to log in command.sh. might need to be last argument to work.')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', 
    help='SGD momentum (default: 0.9)')

parser.add_argument('--nesterov', action='store_true',
        help='use sgd with nesterov instead of adam. (default: adam)')

args = parser.parse_args()

if args.device >= 0:
    device = torch.device(f'cuda:{args.device}')
else:
    device = torch.device('cpu')

os.makedirs(f'{args.dir}/logs', exist_ok=True)
os.makedirs(f'{args.dir}/checkpoints', exist_ok=True)
print(f'seed: {args.seed}')
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
  f.write(' '.join(sys.argv))
  f.write(f'\n\nseed: {args.seed}\n')

dims=[128, 128, 128]
if args.large_patch:
    dims=[160, 192, 128]
if args.xlarge_patch:
    dims=[192, 192, 128]
if args.full_patch:
    dims=[240, 240, 144]

if args.model == 'MonoUNet':
    if args.enhance_feat:
        model = MonoUNet(input_channels=5, upsampling=args.upsampling, instance_norm=args.instance_norm)
        loss = losses.AvgDiceEnhanceLoss(device)
    else:
        if args.cascade_train:
            model = MonoUNet(input_channels=5, upsampling=args.upsampling, instance_norm=args.instance_norm)
        else:
            model = MonoUNet(upsampling=args.upsampling, instance_norm=args.instance_norm)
        loss = losses.AvgDiceLoss()
    if args.loss == 'bce':
        loss = losses.BCELoss()
    if args.loss == 'dicebce':
        loss = losses.DiceBCELoss()

if args.model == 'CascadeNet':
    model = CascadeNet()
    loss = losses.CascadeAvgDiceLoss(coarse_wt_only=args.coarse_wt_only)

if args.model == 'CascadeNetLite':
    model = CascadeNet(lite=True)
    loss = losses.CascadeAvgDiceLoss(coarse_wt_only=args.coarse_wt_only)

if args.model == 'MultiResUNet':
    model = MultiResUNet()
    loss = losses.AvgDiceLoss()

if args.model == 'MultiResVAEReg':
    model = MultiResVAEReg()
    loss = losses.VAEDiceLoss(device)

if args.model == 'VAEReg':
    model = VAEReg()
    loss = losses.VAEDiceLoss(device)

if args.nesterov:
    optimizer = \
        optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
else:
    optimizer = \
        optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

model = model.to(device)
start_epoch = 0

if args.resume:
    print(f'Resume training from {args.resume}')
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])    

if args.pretrain:
    print(f'Begin training from pretrained model {args.pretrain}')
    checkpoint = torch.load(args.pretrain)
    model.load_state_dict(checkpoint["state_dict"])

collate_fn=None
#def collate_fn(batch):
#    bc = bx = by = bz = 0
#    for d in batch:
#        dc, dx, dy, dz = d[0].shape
#        bc = max(bc, dc)
#        bx = max(bx, dx)
#        by = max(by, dy)
#        bz = max(bz, dz)
#    big_d = (bc, bx, by, bz)
#    batch_x = []
#    batch_y = []
#    for d in batch:
#        pad = torch.zeros(big_d)
#        x_shape = d[0].shape
#        pad[:x_shape[0], :x_shape[1], :x_shape[2], :x_shape[3]] = d[0]
#        y_pad = torch.zeros((3, *big_d[1:]))
#        y_shape = d[1].shape
#        y_pad[:y_shape[0], :y_shape[1], :y_shape[2], :y_shape[3]] = d[1]
#        batch_x.append(pad)
#        batch_y.append(y_pad)
#    return torch.stack(batch_x), torch.stack(batch_y)


if args.cross_val:
    filenames=[]
    for (dirpath, dirnames, files) in os.walk(args.data_dir):
        filenames += [os.path.join(dirpath, file) for file in files if '.nii.gz' in file ]

        modes = [sorted([ f for f in filenames if "t1.nii.gz" in f ]),
                      sorted([ f for f in filenames if "t1ce.nii.gz" in f ]),
                      sorted([ f for f in filenames if "t2.nii.gz" in f ]),
                      sorted([ f for f in filenames if "flair.nii.gz" in f ]),
                      sorted([ f for f in filenames if "seg.nii.gz" in f ])
            ]
    joined_files = list(zip(*modes))

    random.shuffle(joined_files)
    split_idx = int(0.8*len(joined_files))
    train_split, val_split = joined_files[:split_idx], joined_files[split_idx:]
    def proc_split(split):
        modes = [[], [], [], []]
        segs = []

        for t1, t1ce, t2, flair, seg in split:
            modes[0].append(t1)
            modes[1].append(t1ce)
            modes[2].append(t2)
            modes[3].append(flair)
            segs.append(seg)
        return modes, segs
    train_modes, train_segs = proc_split(train_split)
    train_data = BraTSTrainDataset(args.data_dir, dims=dims, augment_data=args.augment_data,
            enhance_feat=args.enhance_feat, modes=train_modes, segs=train_segs, )
    trainloader = DataLoader(train_data, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)

    val_modes, val_segs = proc_split(val_split)
    val_data = BraTSTrainDataset(args.data_dir, dims=dims, augment_data=False,
            modes=val_modes, segs=val_segs)
    valloader = DataLoader(val_data, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
elif args.selftrain:
    train_data = BraTSSelfTrainDataset(args.data_dir, model, device, n=args.selftrain_n, dims=dims,   
            augment_data=args.augment_data)
    trainloader = DataLoader(train_data, batch_size=args.batch_size,  
                            shuffle=True, num_workers=args.num_workers)
    val_data = BraTSTrainDataset(args.data_dir, dims=dims, enhance_feat=False, augment_data=False)
    valloader = DataLoader(val_data, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
else:
    # train without cross_val or self-training
    train_data = BraTSTrainDataset(args.data_dir, dims=dims, 
            augment_data=args.augment_data, enhance_feat=args.enhance_feat, throw_no_et_sets=args.throw_no_et_sets)
    trainloader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_fn,
                            shuffle=True, num_workers=args.num_workers)
    val_data = BraTSTrainDataset(args.data_dir, dims=dims, enhance_feat=args.enhance_feat, augment_data=False)
    valloader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn,
                            shuffle=True, num_workers=args.num_workers)

writer = SummaryWriter(log_dir=f'{args.dir}/logs')
scheduler = None

opt = None
if args.swa and args.clr:
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, 2e-4, 2e-8, cycle_momentum=False)
    opt = SWA(scheduler)
elif args.swa:
    swa_lr = ((1 - (args.swa / args.epochs)) ** 0.9) * args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    opt = SWA(optimizer, swa_start=args.swa, swa_freq=1, swa_lr=swa_lr)
elif args.clr or args.eclr:
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, 2e-4, 2e-7, cycle_momentum=False)
    #scheduler = optim.lr_scheduler.CyclicLR(optimizer, 2e-3, 2e-8, cycle_momentum=False)
else:
    # this must occur before giving the optimizer to amp
    lmbda = lambda epoch : (1 - (epoch / args.epochs)) ** 0.9
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

# model has to be on device before passing to amp
if args.mixed_precision:
    # Allow Amp to perform casts as required by the opt_level
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

columns = ['set', 'ep', 'lr', 'loss', 'dice_et', 'dice_wt','dice_tc', \
   'time', 'mem_usage']

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    if args.seedtest:
        model.eval()
        eval_val = validate(model, loss, valloader, device, cascade_train=args.cascade_train, debug=args.debug)
        time_ep = time.time() - time_ep
        memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)

        eval_values = ['eval', epoch + 1, eval_val['loss']] \
          + eval_val['dice'].tolist()\
          + [ time_ep, memory_usage] 

        table = tabulate.tabulate([eval_values], 
                columns, tablefmt="simple", floatfmt="8.4f")
        print(table)

    model.train()

    if args.swa:
        train(model, 
                loss, 
                opt, 
                trainloader, 
                device, 
                cascade_train=arg.cascade_train,
                mixed_precision=args.mixed_precision,
                debug=args.debug)
    else:
         train(model, 
                loss, 
                optimizer, 
                trainloader, 
                device, 
                cascade_train=args.cascade_train,
                mixed_precision=args.mixed_precision,
                debug=args.debug,
                clr=args.clr,
                scheduler=scheduler)
       
    if args.swa and epoch > args.swa:
        opt.swap_swa_sgd()

    if (epoch + 1) % args.save_freq == 0:
        if args.swa:
            opt.bn_update(trainloader, model)
        save_checkpoint(
                f'{args.dir}/checkpoints',
                epoch + 1,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
                msg=args.msg
                )
    
    if (epoch + 1) % args.eval_freq == 0:
        if args.swa:
            opt.bn_update(trainloader, model)
        model.eval()
        if args.cross_val:
            train_val = validate(model, loss, trainloader, 
                    device, cascade_train=args.cascade_train, debug=args.debug)

            writer.add_scalar(f'{args.dir}/logs/loss/train', train_val['loss'], epoch)
            et, wt, tc = train_val['dice']
            writer.add_scalar(f'{args.dir}/logs/dice/train/et', et, epoch)
            writer.add_scalar(f'{args.dir}/logs/dice/train/wt', wt, epoch)
            writer.add_scalar(f'{args.dir}/logs/dice/train/tc', tc, epoch)

            time_ep = time.time() - time_ep
            memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)

            train_values = ['train', epoch + 1, scheduler.get_last_lr(), train_val['loss']] \
              + train_val['dice'].tolist()\
              + [ time_ep, memory_usage] 

            table_train = tabulate.tabulate([train_values], 
                    columns, tablefmt="simple", floatfmt="8.4f")

        eval_val = validate(model, loss, valloader, 
                device, cascade_train=args.cascade_train, debug=args.debug)

        writer.add_scalar(f'{args.dir}/logs/loss/eval', eval_val['loss'], epoch)
        et, wt, tc = eval_val['dice']
        writer.add_scalar(f'{args.dir}/logs/dice/eval/et', et, epoch)
        writer.add_scalar(f'{args.dir}/logs/dice/eval/wt', wt, epoch)
        writer.add_scalar(f'{args.dir}/logs/dice/eval/tc', tc, epoch)

        time_ep = time.time() - time_ep
        memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)

        eval_values = ['eval', epoch + 1, scheduler.get_last_lr(), eval_val['loss']] \
          + eval_val['dice'].tolist()\
          + [ time_ep, memory_usage] 

        table = tabulate.tabulate([eval_values], 
                columns, tablefmt="simple", floatfmt="8.4f")
        if args.cross_val: 
            print(table_train)
        print(table)
   
    writer.flush()
    if not args.swa or not args.clr:
        scheduler.step()
    if args.selftrain:
        trainloader.dataset.annotate()

