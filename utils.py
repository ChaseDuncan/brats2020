import torch
import numpy as np
import json
import os

import nibabel as nib

from configparser import ConfigParser
from torch.utils.data import DataLoader
from losses import (
    dice_score,
    agg_dice_score
    )
import torch.utils.data.sampler as sampler
from tqdm import tqdm

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > .tmp')
    memory_available = [int(x.split()[2]) for x in open('.tmp', 'r').readlines()]
    os.system('rm .tmp')
    return np.argmax(memory_available)

def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)

def save_prediction(src, target, preds, outdir, filename):
    src = src.squeeze().cpu().numpy() 
    target = target.squeeze().cpu().numpy()

    src_npy = src[1, :, :, :]
    img = nib.Nifti1Image(src_npy, np.eye(4))
    nib.save(img, os.path.join(outdir, filename+'.src.nii.gz'))
    # TODO: put this in a loop
    #ground_truth = ['.et_gt.nii.gz', '.tc_gt.nii.gz', '.wt_gt.nii.gz']
    #preds = ['.et_pd.nii.gz', '.tc_pd.nii.gz', '.wt_pd.nii.gz']
    et_npy = target[0, :, :, :]
    et_img = nib.Nifti1Image(et_npy, np.eye(4))
    nib.save(et_img, os.path.join(outdir,filename+'.et_gt.nii.gz'))

    tc_npy = target[1, :, :, :]
    tc_img = nib.Nifti1Image(tc_npy, np.eye(4))
    nib.save(tc_img, os.path.join(outdir, filename+'.tc_gt.nii.gz'))

    wt_npy = target[2, :, :, :]
    wt_img = nib.Nifti1Image(wt_npy, np.eye(4))
    nib.save(wt_img, os.path.join(outdir,filename+'.wt_gt.nii.gz'))

    preds = preds.squeeze().cpu().numpy()

    et_pred = preds[0, :, :, :]
    pred_img = nib.Nifti1Image(et_pred, np.eye(4))
    nib.save(pred_img, os.path.join(outdir, filename+'.et_pd.nii.gz'))

    tc_pred = preds[1, :, :, :]
    pred_img = nib.Nifti1Image(tc_pred, np.eye(4))
    nib.save(pred_img, os.path.join(outdir, filename+'.tc_pd.nii.gz'))

    wt_pred = preds[2, :, :, :]
    pred_img = nib.Nifti1Image(wt_pred, np.eye(4))
    nib.save(pred_img, os.path.join(outdir, filename+'.wt_pd.nii.gz'))


class MRISegConfigParser():
  def __init__(self, config_file):
    config = ConfigParser()
    config.read(config_file)
    self.debug = False 
    self.label_recon = False 

    if config.has_option('data', 'debug'):
      self.debug = config.getboolean('data', 'debug')

    self.deterministic_train = \
        config.getboolean('train_params', 'deterministic_train')
    self.train_split = config.getfloat('train_params', 'train_split')
    self.weight_decay = config.getfloat('train_params', 'weight_decay')
    self.epochs = config.getint('train_params', 'epochs')
    self.data_dir = config.get('data', 'data_dir')
    self.log_dir = config.get('data', 'log_dir')
    self.model_type = config.get('meta', 'model_type')
    self.model_name = config.get('meta', 'model_name')
    self.modes = json.loads(config.get('data', 'modes'))
    self.loss = config.get('meta', 'loss')

    if config.has_option('data', 'dims'):
      self.dims = json.loads(config.get('data', 'dims'))
    if config.has_option('meta', 'label_recon'):
      self.label_recon = config.get_boolean('meta', 'label_recon')


# TODO: clean this up vis a vis checkpoints vs saving model, etc.
def save_model(name, epoch, writer, model, optimizer):
  model_state_dict = {}
  opt_state_dict = {}
  for k, v in model.state_dict().items():
    model_state_dict[k] = v.cpu()
  #for k, v in optimizer.state_dict().items():
  #  opt_state_dict[k] = v.cpu()
  chkpt_dir = 'checkpoints/' + name + '/'
  torch.save({'epoch': epoch,
    #'writer': writer,
    'model_state_dict': model.state_dict(), #model_state_dict,  
    'optimizer_state_dict': optimizer.state_dict()}, chkpt_dir+name)


def load_data(dataset):
  cv_trainloader, cv_testloader = cross_validation(dataset)
  return cv_trainloader[0], cv_testloader[0]


# TODO: currently only using one fold. Either use this or get rid of it.
def cross_validation(dataset, batch_size=1, k = 5):
  num_examples = len(dataset)
  data_indices = np.arange(num_examples)
  np.random.shuffle(data_indices)
  folds = np.array(np.split(data_indices, k))

  cv_trainloader = []
  cv_testloader = []

  for i in range(len(folds)):
    mask = np.zeros(len(folds), dtype=bool)
    mask[i] = True
    train_folds = np.hstack(folds[~mask])
    test_fold = folds[mask][0]
    cv_trainloader.append(DataLoader(dataset,
      batch_size, num_workers=0, sampler=sampler.SubsetRandomSampler(train_folds)))
    cv_testloader.append(DataLoader(dataset,
      batch_size, num_workers=0, sampler=sampler.SubsetRandomSampler(test_fold)))
    return cv_trainloader, cv_testloader


def train(model, loss, optimizer, train_dataloader, device):
  total_loss = 0
  model.train()
  for src, target in tqdm(train_dataloader):
    optimizer.zero_grad()
    src, target = src.to(device, dtype=torch.float),\
        target.to(device, dtype=torch.float)
    output = model(src)
    cur_loss = loss(output, {'target':target, 'src':src})
    total_loss += cur_loss
    cur_loss.backward()
    optimizer.step()
    

def _validate(model, loss, dataloader, device):
    total_loss = 0
    total_dice = 0
    total_dice_agg = 0
    total_examples = 0
    with torch.no_grad():
        model.eval()
        for src, target in tqdm(dataloader):
            src, target = src.to(device, dtype=torch.float),\
                target.to(device, dtype=torch.float)
            total_examples+=src.size()[0]
            output = model(src)
            total_loss += loss(output, {'target':target, 'src':src}) 
            total_dice += dice_score(output, target)
            total_dice_agg += agg_dice_score(output, target)
            ####### 
            # CascadeNet
            #
            #average_seg = 0.5*(output['deconv'] + output['biline'])
            #total_dice += dice_score(average_seg, target)
            #total_dice_agg += agg_dice_score(average_seg, target)
        
        avg_dice = total_dice / total_examples
        avg_dice_agg = total_dice_agg / total_examples 
        avg_loss = total_loss /  total_examples
        return avg_dice, avg_dice_agg, avg_loss

def validate(model, loss, trainloader, device):
  train_dice, train_dice_agg, train_loss =\
      _validate(model, loss, trainloader, device)
  test_dice = None
  test_dice_agg = None
  test_loss = None

  if testloader:
    test_dice, test_dice_agg, test_loss =\
        _validate(model, loss, testloader, device, True)

  return {'train_dice':train_dice, 'train_dice_agg':train_dice_agg, 
          'train_loss':train_loss, 'test_dice':test_dice, 
          'test_dice_agg':test_dice_agg, 'test_loss':test_loss}

