import configparser

import h5py
import torch

from environment.utils import GridTransform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(10)

cfg = configparser.ConfigParser()
cfg.read("data/config.ini")
H5FILE_BIN = "data/" + cfg["Preprocess"]["FILE_PREFIX"] + "bin.h5"

fh5_bin = h5py.File(H5FILE_BIN, "r")
voxdims_all = fh5_bin["voxdims_all"][()]

idx = 6

gland = torch.tensor(fh5_bin["/gland_%04d" % idx][()], dtype=torch.bool, device=device)
targets = torch.tensor(
    fh5_bin["/targets_%04d" % idx][()], dtype=torch.uint8, device=device
)
num_t = targets.max()
gland = gland[None, None].repeat(num_t, 1, 1, 1, 1)
target = torch.stack([targets == (i + 1) for i in range(num_t)]).unsqueeze(1)

volume = torch.concat([gland, target], dim=1).type(torch.float32)[...,:-10,:]  # reduce x for debugging


random_transform = GridTransform(grid_size=[7,8,4], interp_type='t-conv', volsize=[volume.shape[i] for i in [3,2,4]], batch_size=volume.shape[0], device=device)
random_transform.generate_random_transform()
transformed_volume = random_transform.warp(volume)

#"""debug
import SimpleITK as sitk
threshold = 0.45    
for b in range(volume.shape[0]):
    sitk.WriteImage(sitk.GetImageFromArray((volume[b,0,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_gland.nii'%b)
    sitk.WriteImage(sitk.GetImageFromArray((volume[b,1,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_target.nii'%b)
    sitk.WriteImage(sitk.GetImageFromArray((transformed_volume[b,0,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_gland_transformed.nii'%b)
    sitk.WriteImage(sitk.GetImageFromArray((transformed_volume[b,1,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_target_transformed.nii'%b)
print('volumes saved.')
#"""
