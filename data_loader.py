import os

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from abc import abstractmethod
import random
from tqdm import tqdm

def shuffle_split_dataset(data_dir, split_idx):
    def _proc_split(split):
        modes = [[], [], [], []]
        segs = []

        if len(split[0]) == 5:
            for t1, t1ce, t2, flair, seg in split:
                modes[0].append(t1)
                modes[1].append(t1ce)
                modes[2].append(t2)
                modes[3].append(flair)
                segs.append(seg)
        else:
            for t1, t1ce, t2, flair in split:
                modes[0].append(t1)
                modes[1].append(t1ce)
                modes[2].append(t2)
                modes[3].append(flair)

        return modes, segs

    filenames=[]
    for (dirpath, dirnames, files) in os.walk(data_dir):
        filenames += [os.path.join(dirpath, file) for file in files if '.nii.gz' in file ]
    modes = [sorted([ f for f in filenames if "t1.nii.gz" in f ]),
              sorted([ f for f in filenames if "t1ce.nii.gz" in f ]),
              sorted([ f for f in filenames if "t2.nii.gz" in f ]),
              sorted([ f for f in filenames if "flair.nii.gz" in f ]),
                sorted([ f for f in filenames if "seg.nii.gz" in f ])

        ]
    modes = [mode for mode in modes if len(mode) > 0 ]
    joined_files = list(zip(*modes))
    # this could be an option...
    random.shuffle(joined_files)
    train_split, val_split = joined_files[:split_idx], joined_files[split_idx:]
    return _proc_split(train_split), _proc_split(val_split)

class BraTSDataset(Dataset):
    def __init__(self, data_dir, dims=[240, 240, 155], modes=None, segs=None):
        self.x_off = 0
        self.y_off = 0
        self.z_off = 0
        self.dims=dims
        
        filenames = []
        # should have a conditional here to skip this is modes and segs are not None
        for (dirpath, dirnames, files) in os.walk(data_dir):
            filenames += [os.path.join(dirpath, file) for file in files if '.nii.gz' in file ]

        self.modes = [sorted([ f for f in filenames if "t1.nii.gz" in f ]),
                      sorted([ f for f in filenames if "t1ce.nii.gz" in f ]),
                      sorted([ f for f in filenames if "t2.nii.gz" in f ]),
                      sorted([ f for f in filenames if "flair.nii.gz" in f ])
            ]

        self.segs = sorted([ f for f in filenames if "seg.nii.gz" in f ])

        if modes:
            self.modes = modes
        if segs:
            self.segs = segs



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

    def _transform_data(self, d, seg_mat=False, shift_and_scale=False):
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

        #img = np.zeros(img.shape)
        #img[120, 120, 50] = 1
        #print(np.where(img > 0))

        self.x_off = (img.shape[0] - self.dims[0]) // 2
        self.y_off = (img.shape[1] - self.dims[1]) // 2
        self.z_off = (img.shape[2] - self.dims[2]) // 2

        img = img[self.x_off:img.shape[0]-self.x_off,
              self.y_off:img.shape[1]-self.y_off,
              self.z_off:img.shape[2]-self.z_off]
        #dims = self.dims

        #x_off = int((240 - dims[0]) / 2)
        #y_off = int((240 - dims[1]) / 2)
        #z_off = int((155 - dims[2]) / 2)
        #m = nn.ConstantPad3d((z_off+1, z_off, y_off, y_off, x_off, x_off), 0)
        #img = m(img)
        #print(np.where(img > 0))
        #import sys; sys.exit()
        #et = m(output[0, 0, :, :, :])
        #wt = m(output[0, 1, :, :, :])
        #tc = m(output[0, 2, :, :, :])

        # don't standardized the segmentations
        if seg_mat:
            return img

        #img_trans = self.min_max_normalize(img)
        return self.standardize(img, shift_and_scale=shift_and_scale)

    def min_max_normalize(self, d):
        d = (d - np.min(d)) / (np.max(d) - np.min(d))
        return d

    def standardize(self, d, shift_and_scale=False):
        # H length list of WxD arrays of booleans where
        # the i, (j, k) is True if d[i, j, k] > 0
        nonzero_masks = [i != 0 for i in d]
        brain_mask = np.zeros(d.shape, dtype=bool)

        for i in range(len(nonzero_masks)):
            brain_mask[i, :, :] = brain_mask[i, :, :] | nonzero_masks[i]
        
        mean = d[brain_mask].mean()
        std = d[brain_mask].std()
        # now normalize each modality with its mean and standard deviation (computed within the brain mask)
        stan = (d - mean) / (std + 1e-8)
        if shift_and_scale:
            stan = stan + self.shft
            stan = stan*self.scal 

        stan[brain_mask == False] = 0
        return stan

    @abstractmethod
    def __getitem__(self, idx):
        pass

        
class BraTSTrainDataset(BraTSDataset):
    def __init__(self, data_dir, 
        dims=[240, 240, 155], 
        augment_data = True, throw_no_et_sets=False,
        clinical_segs=True, enhance_feat=False, 
        modes=None, segs=None):
        BraTSDataset.__init__(self, data_dir, dims, modes=modes, segs=segs)
        self.clinical_segs = clinical_segs
        self.enhance_feat=enhance_feat

        self.augment_data = augment_data

        # randomly mirror along axis
        self.mirror = False
        self.axis = np.random.choice([0, 1, 2], 1)[0]
        
        self.shft = None
        self.scal = None

        if throw_no_et_sets:
            print('Removing datasets with no enhancing tumor.')
            segs_temp = []
            mode_temp = [[],[],[],[]]
            kicked = 0
            for seg, m1, m2, m3, m4 in zip(self.segs, *self.modes):
                img = nib.load(seg).get_fdata()
                if len(np.where(img==4)[0]) != 0:
                    segs_temp.append(seg)
                    mode_temp[0].append(m1)
                    mode_temp[1].append(m2)
                    mode_temp[2].append(m3)
                    mode_temp[3].append(m4)
                else:
                    kicked+=1
                    
            print(f'{kicked + 1} datasets removed.')
            self.segs = segs_temp
            self.modes = mode_temp

    def _load_images(self, idx):
        images = []
        for m in self.modes:
            image = nib.load(m[idx])
            images.append(image)
            header = image.header
        return images, header

    def _transform_data(self, image, seg_mat=False, shift_and_scale=False):
        img_trans = BraTSDataset._transform_data(self, image, 
                seg_mat=seg_mat, shift_and_scale=shift_and_scale)
        if self.mirror:
            # not sure the copy is needed here
            img_trans = np.flip(img_trans, self.axis).copy()

        return img_trans

    def __getitem__(self, idx):
        # mirror sample? if so which dimension
        shift_and_scale=False
        if self.augment_data:
            self.shft = np.random.uniform(-0.1, 0.1, self.dims)
            self.scal = np.random.uniform(0.9, 1.1, self.dims)
            shift_and_scale = True

        if np.random.uniform() > 0.5: 
            self.mirror = True
            self.axis = np.random.choice([0, 1, 2], 1)[0]

        # header data should be handled in preprocessing, not here
        images, header = self._load_images(idx) 
        images = [self._transform_data(image, shift_and_scale=shift_and_scale) for image in images]  
        images = torch.from_numpy(np.stack(images))
        
        if self.enhance_feat:
            # t1 idx: 0 t1ce idx: 1
            e = images[1] / (images[0] + 1e-8)
            images = torch.cat([images, e.unsqueeze(0)])

        target = []
        # get this out of here
        if self.segs:
            seg = nib.load(self.segs[idx])
            seg = self._transform_data(seg, seg_mat=True)
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

        return images, target


class BraTSSelfTrainDataset(BraTSTrainDataset):
    ''' Dataset class for self training.  '''
    def __init__(self, data_dir,  model, device,
            unsupervised_data_dir='/shared/mrfil-data/cddunca2/Task01_BrainTumour/partitioned-by-mode/',  
            n=50, dims=[240, 240, 155],
            augment_data = True, modes=None, segs=None):
        BraTSTrainDataset.__init__(self, data_dir, dims, 
                augment_data = augment_data, enhance_feat=False,
                modes=modes, segs=segs)
        self.orig_segs = self.segs.copy()
        self.unsupervised_data_dir = unsupervised_data_dir
        self.model = model
        self.device = device
        
        (st_modes, _), (_, _) = shuffle_split_dataset(unsupervised_data_dir, n)
        # shuffle data and select first n 
        unsupervised_data = BraTSAnnotationDataset(unsupervised_data_dir, 
                modes = st_modes, 
                dims=dims, 
                enhance_feat=False)
        # batch size > 1 breaks something 
        #self.dataloader = DataLoader(unsupervised_data, batch_size=5)
        self.dataloader = DataLoader(unsupervised_data, batch_size=1)
        # this will cause problems for multiple processes
        self.annotations_dir = '/dev/shm/tmp/selftrain/'
        os.makedirs(self.annotations_dir, exist_ok=True)
         
        self.modes = sorted([self.modes[i] + st_modes[i] for i in range(len(self.modes))])
        self.segs = sorted(self.segs+ self.annotate())

    def annotate(self):
        segs = []
        with torch.no_grad():
            self.model.eval()
            for d in tqdm(self.dataloader):
                src = d['data'].to(self.device, dtype=torch.float)
                output = self.model(src)

                x_off = int((240 - self.dims[0]) / 2)
                y_off = int((240 - self.dims[1]) / 2)
                z_off = int((155 - self.dims[2]) / 2)

                m = nn.ConstantPad3d((z_off, z_off, y_off, y_off, x_off, x_off), 0)

                et = m(output[0, 0, :, :, :])
                wt = m(output[0, 1, :, :, :])
                tc = m(output[0, 2, :, :, :])
                
                label = torch.zeros((240, 240, 155))
                label[torch.where(wt > 0.50)] = 2
                label[torch.where(tc > 0.50)] = 1
                label[torch.where(et > 0.50)] = 4
                
                orig_header = nib.load(d['file'][0][0]).header
                aff = orig_header.get_qform()
                img = nib.Nifti1Image(label.numpy(), aff, header=orig_header)
                seg_path = os.path.join(self.annotations_dir, f'{d["patient"][0]}_seg.nii.gz')
                img.to_filename(seg_path)
                segs.append(seg_path)
        return segs

###### BraTSSelfTrainDataset

class BraTSAnnotationDataset(BraTSDataset):
    def __init__(self, data_dir,  
        dims=[240, 240, 155], augment_data = True, clinical_segs=True,
        enhance_feat=False, modes=None):
        BraTSDataset.__init__(self, data_dir, modes=modes, dims=dims)
        self.enhance_feat=enhance_feat

    def _patient(self, f):
        return f.split('/')[-2]

    def __getitem__(self, idx):
        # header data should be handled in preprocessing, not here
        images, header = self._load_images(idx) 
        data = [torch.from_numpy(self._transform_data(img)) for img in images]
        if self.enhance_feat:
            # t1 idx: 0 t1ce idx: 1
            e = data[1] / (data[0] + 1e-32)
            e[e<0] = 0 
            e[e>0] = 1
            data.append(e) 

        src = torch.stack(data)

        ## this has to be on for outputing segmentations
        #  otherwise the metadata won't be correct
        # ideally this should be done in preprocessing or 
        # something not here.
        return {'patient': self._patient(self.modes[0][idx]), 
                'data': src, 
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

