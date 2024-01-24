import numpy as np 
from matplotlib import pyplot as plt 
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader, RandomSampler 
import pandas as pd 
import os 
import h5py
from networks.networks import * 
from utils.utils import * 
import argparse 

#Arg parse functions 
parser = argparse.ArgumentParser(prog='train',
                                description="Train RL agent See list of available arguments for more info.")

parser.add_argument('--log_dir',
                    '--log',
                    metavar='log_dir',
                    type=str,
                    action='store',
                    default='Timestep_SL_v3',
                    help='Log dir to save results to')

parser.add_argument('--loss_fn',
                    '--loss',
                    metavar='loss_fn',
                    type=str,
                    action='store',
                    default='MSE',
                    help='Which loss fn to use')

parser.add_argument('--finetune',
                    '--ft',
                    metavar='finetune',
                    type=bool,
                    action='store',
                    default=False,
                    help='Whether or not to finetune')

parser.add_argument('--ft_modelpath',
                    '--ft_mp',
                    metavar='ft_modelpath',
                    type=str,
                    action='store',
                    default='/raid/candi/Iani/Biopsy_RL/Supervised_code/Timestep_SL_v2/best_val_model.pth',
                    help='Whether or not to finetune')

if __name__ == '__main__':

    ### PATH_FILES
    PS_PATH = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality'
    #LABELS_PATH = '/Users/ianijirahmae/Documents/PhD_project/Biopsy_RL/grid_labels_all.h5'
    
    ### Path files for PT
    args = parser.parse_args()
    #PS_PATH = '/raid/candi/Iani/MRes_project/Reinforcement Learning/DATASETS/'
    LABELS_PATH = '/raid/candi/Iani/Biopsy_RL/action_labels.h5'
    SAVE_FOLDER = args.log_dir
    LOSS_FN = args.loss_fn

    FINETUNE = args.finetune

    ## Training script 
    train_ds = TimeStep_data(PS_PATH, LABELS_PATH, mode = 'train', finetune = FINETUNE)
    val_ds = TimeStep_data(PS_PATH, LABELS_PATH, mode = 'val', finetune = FINETUNE)

    ### Define dataloaders for train and validation 
    train_dl = DataLoader(train_ds, batch_size = 32, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size = 8, shuffle = False)

    ### Define network
    UNet_model = ImitationNetwork()

    if FINETUNE: 
        print(f'Fine-tuning model using first frames of every dataset!')
        MODEL_PATH = args.ft_modelpath
        UNet_model.load_state_dict(torch.load(MODEL_PATH))
    

    ### Train script 
    all_loss_train, all_loss_val, all_acc_train, all_acc_val  = train_pertimestep(UNet_model, train_dl, val_dl, \
        num_epochs = 10000, use_cuda = True, save_folder = SAVE_FOLDER, loss_fn_str = LOSS_FN) 

    print('Chicken')
