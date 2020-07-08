import os

import numpy as np
import nibabel as nib
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from batchgenerators.utilities.file_and_folder_operations import *

class BraTSValidation(Dataset):
    def __init__(self, data_dir):
        self.patients = get_list_of_patients(data_dir)        
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        data, metadata = load_patient(os.path.join(self.data_dir, self.patients[idx]))
        patient_id = self.patients[idx].split('/')[-1]
        metadata['patient_id'] = patient_id
        return data, metadata

DEBUG=False
#DEBUG=True

def get_list_of_patients(preprocessed_data_folder):
    npy_files = subfiles(preprocessed_data_folder, suffix=".npy", join=True)
    # remove npy file extension
    patients = [i[:-4] for i in npy_files]
    return patients


def load_patient(patient):
    data = np.load(patient + ".npy")
    metadata = load_pickle(patient + ".pkl")
    return data, metadata

class BraTSDataset(Dataset):
    def __init__(self, data_dir, modes=['t1', 't1ce', 't2', 'flair'], 
        dims=[240, 240, 155], augment_data = False, clinical_segs=True):

        self.clinical_segs = True
        self.x_off = 0
        self.y_off = 0
        self.z_off = 0
        self.dims=dims

        filenames = []
        for (dirpath, dirnames, files) in os.walk(data_dir):
          filenames += [os.path.join(dirpath, file) for file in files if '.nii.gz' in file ]

        self.modes = [sorted([ f for f in filenames if "t1.nii.gz" in f ]),
                      sorted([ f for f in filenames if "t1ce.nii.gz" in f ]),
                      sorted([ f for f in filenames if "t2.nii.gz" in f ]),
                      sorted([ f for f in filenames if "flair.nii.gz" in f ])
            ]

        self.segs = sorted([ f for f in filenames if "seg.nii.gz" in f ])

        self.augment_data = augment_data

        # randomly flip along axis
        self.axis = None

        # TODO: random flip isn't working
        if self.augment_data:
          if a > 0.5:
            self.axis = np.random.choice([0, 1, 2], 1)[0]
        # for debugging
        self.src = None
        self.target = None

    def __len__(self):
        # return size of dataset
        return max([len(self.modes[i]) for i in range(len(self.modes))])

    def data_aug(self, brain):
        if self.axis:
            brain = np.flip(brain, self.axis).copy()
        shift_brain = brain + torch.Tensor(np.random.uniform(-0.1, 0.1, brain.shape)).double().cuda()
        scale_brain = shift_brain*torch.Tensor(np.random.uniform(0.9, 1.1, brain.shape)).double().cuda()
        return scale_brain


    # TODO: mask brain
    # changing data type in the function is stupid
    def std_normalize(self, d):
      ''' Subtract mean and divide by standard deviation of the image.'''
      d = torch.from_numpy(d)
      d_mean = torch.mean(d)
      means = [d_mean]*d.shape[0]
      d_std = torch.std(d)
      stds = [d_std]*d.shape[0]
      d_trans = TF.normalize(d, means, stds).cuda()
      return d_trans

    def _transform_data(self, d):
        img = nib.load(d).get_fdata()
        x, y, z = img.shape
        add_x = x % 2 
        add_y = y % 2 
        add_z = z % 2 
        npad = ((0, add_x),
                (0, add_y),
                (0, add_z))
        img = np.pad(img, pad_width=npad, mode='constant', constant_values=0)
        self.x_off = (img.shape[0] - self.dims[0]) // 2
        self.y_off = (img.shape[1] - self.dims[1]) // 2
        self.z_off = (img.shape[2] - self.dims[2]) // 2

        img = img[self.x_off:img.shape[0]-self.x_off,
              self.y_off:img.shape[1]-self.y_off,
              self.z_off:img.shape[2]-self.z_off]
        img_trans = self.min_max_normalize(img)
        #img_trans = self.std_normalize(img)

        if self.augment_data:
            img_trans = self.data_aug(img_trans)
        return img_trans


    def min_max_normalize(self, d):
        # TODO: changing data type in the function is stupid
        d = torch.from_numpy(d)
        d = (d - d.min()) / (d.max() - d.min())
        return d


    def __getitem__(self, idx):
        if DEBUG and self.src != None and self.target != None:
            return self.src, self.target
        elif DEBUG:
            idx=0

        data = [self._transform_data(m[idx]) for m in self.modes]
        src = torch.stack(data)

        target = []
        if self.segs:
            seg = nib.load(self.segs[idx]).get_fdata()
            x, y, z = seg.shape
            add_x = x % 2 
            add_y = y % 2 
            add_z = z % 2 
            npad = ((0, add_x),
                    (0, add_y),
                    (0, add_z))
            seg = np.pad(seg, pad_width=npad, mode='constant', constant_values=0)
            self.x_off = (seg.shape[0] - self.dims[0]) // 2
            self.y_off = (seg.shape[1] - self.dims[1]) // 2
            self.z_off = (seg.shape[2] - self.dims[2]) // 2
    
            seg = seg[self.x_off:seg.shape[0]-self.x_off,
              self.y_off:seg.shape[1]-self.y_off,
              self.z_off:seg.shape[2]-self.z_off]

            if self.axis:
                seg = np.flip(seg, axis)

            segs = []
            # TODO: Wrap in a loop.
            if self.clinical_segs:
                # enhancing tumor
                seg_et = np.zeros(seg.shape)
                seg_et[np.where(seg==4)] = 1
                segs.append(seg_et)

                # whole tumor
                seg_wt = np.zeros(seg.shape)
                seg_wt[np.where(seg==1)] = 1
                seg_wt[np.where(seg==2)] = 1
                seg_wt[np.where(seg==4)] = 1
                segs.append(seg_wt)
               
                # tumor core
                seg_tc = np.zeros(seg.shape)
                seg_tc[np.where(seg==1)] = 1
                seg_tc[np.where(seg==4)] = 1
                segs.append(seg_tc)

                target = torch.from_numpy(np.stack(segs))
            else:
                # necrotic/non-enhancing tumor
                seg_ncr_net = np.zeros(seg.shape)
                seg_ncr_net[np.where(seg==1)] = 1
                segs.append(seg_ncr_net)
                
                # edema
                seg_ed = np.zeros(seg.shape)
                seg_ed[np.where(seg==2)] = 1
                segs.append(seg_ed)
                
                # enhancing tumor
                seg_et = np.zeros(seg.shape)
                seg_et[np.where(seg==4)] = 1
                segs.append(seg_et)

                target = torch.from_numpy(np.stack(segs))

        if '_t1' in self.modes[0][idx] and not self.segs:
            target = self.modes[0][idx].replace('_t1', '')
        if DEBUG:
            self.src = src
            self.target = target
        return src, target
