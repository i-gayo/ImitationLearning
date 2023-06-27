

import os
import sys
import configparser

import h5py

from environment.biopsy_env import TPBEnv


cfg = configparser.ConfigParser()
cfg.read('config.ini')
TARGET_VOLUME_LOW = eval(cfg['Preprocess']['TARGET_VOLUME_LOW'])
H5FILE_BIN = cfg['Preprocess']['FILE_PREFIX'] + 'bin.h5'


def main():
    
    fh5_bin = h5py.File(H5FILE_BIN, 'r')
    # filenames_all = [b.decode("utf-8") for b in fh5_bin_g['filenames_all'][()]]
    voxdims_all = fh5_bin['voxdims_all'][()]

    for idx, voxdims in enumerate(voxdims_all):

        gland = fh5_bin['/gland_%04d' % idx][()]
        targets = fh5_bin['/targets_%04d' % idx][()]

        tpb_env = TPBEnv() # create an biopsy environment, with default guide and transition

        episodes = tpb_env.generate_episodes()

        # save episodes to files




if __name__ == "__main__":
    main()
