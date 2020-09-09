
import os
import nibabel as nib

filenames=[]
unsupervised_data_dir='/shared/mrfil-data/cddunca2/Task01_BrainTumour/imagesTr'
for (dirpath, dirnames, files) in os.walk(unsupervised_data_dir):
    filenames += [os.path.join(dirpath, file) for file in files if '.nii.gz' in file ]

out_dir ='/shared/mrfil-data/cddunca2/Task01_BrainTumour/partitioned-by-mode'
for decath_file in filenames:
    patient = decath_file.split('/')[-1].split('.')[0]
    image = nib.load(decath_file)
    image_tns = image.get_fdata()
    orig_header = image.header
    aff = orig_header.get_qform()
    # flair t1 t1ce t2 
    modes = ["flair.nii.gz", "t1.nii.gz", "t1ce.nii.gz", "t2.nii.gz"]
    for i, mode in enumerate(modes):
        img_mat = image_tns[:, :, :, i]
        img = nib.Nifti1Image(img_mat, aff, header=orig_header)
        os.makedirs(f'{out_dir}/{patient}', exist_ok=True)
        img.to_filename(os.path.join(out_dir, f'{patient}/{patient}_{mode}'))


