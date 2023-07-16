
import configparser

import h5py
import torch

from environment.biopsy_env import TPBEnv


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


cfg = configparser.ConfigParser()
cfg.read("data/config.ini")
H5FILE_BIN = "data/" + cfg["Preprocess"]["FILE_PREFIX"] + "bin.h5"


def main():
    fh5_bin = h5py.File(H5FILE_BIN, "r")
    # filenames_all = [b.decode("utf-8") for b in fh5_bin_g['filenames_all'][()]]
    voxdims_all = fh5_bin["voxdims_all"][()]

    for idx, voxdims in enumerate(voxdims_all):
        # debug: idx = 10 # 6

        gland = torch.tensor(
            fh5_bin["/gland_%04d" % idx][()], dtype=torch.bool, device=device
        )
        targets = torch.tensor(
            fh5_bin["/targets_%04d" % idx][()], dtype=torch.uint8, device=device
        )
        # debug: 
        gland, targets = gland[:,:-35,:], targets[:,:-35,:]

        num_t = targets.max()
        tpb_envs = TPBEnv(
            gland=gland[None, None].repeat(num_t, 1, 1, 1, 1),
            target=torch.stack([targets == (i + 1) for i in range(num_t)]).unsqueeze(1),
            voxdims=[voxdims.tolist()] * num_t,
        )  # create a predefined biopsy environment

        episodes = tpb_envs.run()
        
        print("Simulation done for all data in the No.%d batch." % idx)

        #TODO: save episodes to files


if __name__ == "__main__":
    main()
