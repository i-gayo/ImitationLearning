import configparser

import h5py
import torch

from environment.utils import grid_transform

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


random_transform = grid_transform(grid_size=[7,8,4], volsize=[volume.shape[i] for i in [3,2,4]], batch_size=volume.shape[0], device=device)
random_transform = random_transform.generate_random_transform()
transformed_volume = random_transform.warp(volume)