import os

import numpy as np
import nibabel as nib
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from abc import abstractmethod

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
    def __init__(self, data_dir, modes=['t1', 't1ce', 't2', 'flair'], dims=[240, 240, 155]):
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


    def __len__(self):
        # return size of dataset
        return max([len(self.modes[i]) for i in range(len(self.modes))])

    def _load_images(self, idx):
        images = []
        for m in self.modes:
            image = nib.load(m[idx])
            images.append(image)
            header = image.header

        return images, header

    def _transform_data(self, d):
        img = d.get_fdata()
        x, y, z = img.shape
        add_x = x % 2 
        add_y = y % 2 
        add_z = z % 2 
        npad = ((0, add_x),
                (0, add_y),
                (0, add_z))
        img = np.pad(img, 
                pad_width=npad, 
                mode='constant', 
                constant_values=0)
        self.x_off = (img.shape[0] - self.dims[0]) // 2
        self.y_off = (img.shape[1] - self.dims[1]) // 2
        self.z_off = (img.shape[2] - self.dims[2]) // 2

        img = img[self.x_off:img.shape[0]-self.x_off,
              self.y_off:img.shape[1]-self.y_off,
              self.z_off:img.shape[2]-self.z_off]

        img_trans = self.min_max_normalize(img)

        return img_trans


    def min_max_normalize(self, d):
        d = (d - np.min(d)) / (np.max(d) - np.min(d))
        return d


    @abstractmethod
    def __getitem__(self, idx):
        pass

class BraTSTrainDataset(BraTSDataset):
    def __init__(self, data_dir, modes=['t1', 't1ce', 't2', 'flair'], 
        dims=[240, 240, 155], augment_data = True, clinical_segs=True):
        BraTSDataset.__init__(self, data_dir, modes, dims)
        self.clinical_segs = clinical_segs

        self.augment_data = augment_data

        # randomly mirror along axis
        self.mirror = False
        self.axis = np.random.choice([0, 1, 2], 1)[0]

    def data_aug(self, brain):
        shift_brain = brain + np.random.uniform(-0.1, 0.1, brain.shape)
        scale_brain = shift_brain*np.random.uniform(0.9, 1.1, brain.shape)
        return scale_brain

    def _load_images(self, idx):
        images = []
        for m in self.modes:
            image = nib.load(m[idx])
            images.append(image)

            header = image.header
        return images, header
    

    def _transform_data(self, d):
        img_trans = BraTSDataset._transform_data(self, d)
        self.mirror=False
        if self.augment_data:
            img_trans = self.data_aug(img_trans)
            if np.random.uniform() > 0.5:
                img_trans = np.flip(img_trans, self.axis).copy()
                self.mirror=True
        return img_trans

    def __getitem__(self, idx):
        # header data should be handled in preprocessing, not here
        images, header = self._load_images(idx) 
        data = [torch.from_numpy(self._transform_data(img)) for img in images]
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

            if self.augment_data and self.mirror:
                seg = np.flip(seg, self.axis)

            segs = []

            if self.clinical_segs:
                # enhancing tumor
                seg_et = np.zeros(seg.shape)
                seg_et[np.where(seg==4)] = 1
                segs.append(seg_et)

                # whole tumor
                seg_wt = np.zeros(seg.shape)
                seg_wt[np.where(seg > 0)] = 1
                segs.append(seg_wt)
               
                # tumor core
                seg_tc = np.zeros(seg.shape)
                seg_tc[np.where(seg==1) or np.where(seg==4)] = 1
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
        return src, target

class BraTSAnnotationDataset(BraTSDataset):
    def __init__(self, data_dir, modes=['t1', 't1ce', 't2', 'flair'], 
        dims=[240, 240, 155], augment_data = True, clinical_segs=True):
        BraTSDataset.__init__(self, data_dir, modes, dims)

    def _patient(self, f):
        return f.split('/')[-2]

    def __getitem__(self, idx):
        # header data should be handled in preprocessing, not here
        images, header = self._load_images(idx) 
        data = [torch.from_numpy(self._transform_data(img)) for img in images]
        src = torch.stack(data)

        ## this has to be on for outputing segmentations
        #  otherwise the metadata won't be correct

        return {'patient': self._patient(self.modes[0][idx]), 
                'data': src, 
                # this should be done in preprocessing or something not here
                'qoffsets': [
                    header['qoffset_x'],
                    header['qoffset_y'],
                    header['qoffset_z']
                    ],
                'srows': [
                    header['srow_x'],
                    header['srow_y'],
                    header['srow_z'],
                    ],
                'file': [self.modes[0][idx]]
                }

