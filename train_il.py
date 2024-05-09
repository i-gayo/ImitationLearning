import numpy as np 
from matplotlib import pyplot as plt 
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader, RandomSampler 
import pandas as pd 
import os 
import h5py
import argparse 

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import CnnPolicy
from supersuit import frame_stack_v1
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gym

#Processes for multi-env processing 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
#from utils import * # _data import *
from utils.rl_utils import *
from stable_baselines3.common.policies import ActorCriticPolicy

#Importing module functions 
from utils.Prostate_dataloader import *
from envs.Biopsy_env import TemplateGuidedBiopsy, TemplateGuidedBiopsy_single
import random 
from networks.networks_il import *
from utils.il_utils import * 

#Arg parse functions 
parser = argparse.ArgumentParser(prog='train',
                                description="Train RL agent See list of available arguments for more info.")

parser.add_argument('--log_dir',
                    '--log',
                    metavar='log_dir',
                    type=str,
                    action='store',
                    # default='IL_agent_NEW',
                    default='Logs/IL_agent_NEW_4',
                    help='Log dir to save results to')

parser.add_argument('--loss_fn',
                    '--loss',
                    metavar='loss_fn',
                    type=str,
                    action='store',
                    default='MSE',
                    help='Which loss fn to use')

parser.add_argument('--clip_actions',
                    '--clip',
                    metavar='clip_actions',
                    type=bool,
                    action='store',
                    default=True,
                    help='Whether or not to clip actions during training /t esting')

parser.add_argument('--single_patient',
                    '--sp',
                    metavar='single_patient',
                    type=bool,
                    action='store',
                    default=False,
                    help='Whether or not to debug use single patient only')

parser.add_argument('--data_mode',
                    '--dm',
                    metavar='data_mode',
                    type=str,
                    action='store',
                    default='GS',
                    help='Which training data to use : c2l, n2n or l2l')

parser.add_argument('--train_strategy',
                    '--ts',
                    metavar='train_strategy',
                    type=str,
                    action='store',
                    default='individual',
                    help='Which training strategy to use : individual, subsample or sequential')

parser.add_argument('--using_rl',
                    metavar='using_rl',
                    type=str,
                    action='store',
                    default='True',
                    help='Whether to use RL networks or not')

parser.add_argument('--feature_extractor',
                    metavar='feature_extractor',
                    type=str,
                    action='store',
                    default='imitationconv',
                    help='Which feature extractor to use')

parser.add_argument('--pretrain_path',
                    metavar='pretrain_path',
                    type=str,
                    action='store',
                    default='sequential',
                    help='Which pretraining model to use')

parser.add_argument('--pretrain',
                    metavar='pretrain',
                    type=str,
                    action='store',
                    default='False',
                    help='Whether to pre-train or not')

parser.add_argument('--num_timesteps',
                    metavar='num_timesteps',
                    type=str,
                    action='store',
                    default='3',
                    help='How many timestep channels to use')

if __name__ =='__main__':

    test_label = 1
    PS_PATH = 'ProstateDataset'
    CSV_PATH = 'ProstateDataset/patient_data_multiple_lesions.csv'
    args = parser.parse_args()
    data_mode = args.data_mode
    # load dataset: 
    if data_mode == 'GS':
        LABELS_PATH = 'DATA/NEW_GS.h5'
    else:
        LABELS_PATH = 'DATA/NEW_WACKY.h5'
    hf = h5py.File(LABELS_PATH, 'r')
    
    # Parse arguments 
    print(f"Using data mode : {data_mode}")
    training_strategy = args.train_strategy
    TRAIN_MODE = 'train'
    USING_RL = (args.using_rl == 'True') or (args.using_rl == 'true')
    use_cuda = torch.cuda.is_available()
    FEATURE_EXTRACTOR = args.feature_extractor
    device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device_cuda}')
    clip_actions = args.clip_actions
    print(f'Clipped actions : {clip_actions}')
    LOSS_FN = args.loss_fn
    print(f"Loss fn : {LOSS_FN}")

    
    # SAVE LOG FILE NAME WITH PARAMETERS CHOSEN 
    if args.log_dir == 'IL_agent': 
        #ie remains unchanged
        if USING_RL:
            print(f"Using RL: {USING_RL}")
            LOG_DIR = 'IL_' + training_strategy + '_RL_'
        else:
            print(f"Not using RL:")
            LOG_DIR = 'IL_' + training_strategy + '_NRL_' + FEATURE_EXTRACTOR
    else:
        LOG_DIR = args.log_dir  
        os.makedirs(LOG_DIR, exist_ok=True) 
    print(f"Saving all results/models to folder : {LOG_DIR}")
    
    # Pretraining paths 
    PRETRAIN = (args.pretrain == 'True')
    if args.pretrain_path == 'sequential':
        PRETRAIN_PATH = 'Logs/IL_agent_NEW_4/best_val_model.pth'
    else:
        PRETRAIN_PATH = 'Logs/IL_agent_NEW_4/best_val_model.pth'
        
    FEATURE_EXTRACTOR = args.feature_extractor
    num_timesteps = int(args.num_timesteps)
    
    # SETTING UP RL ENVIRONMENT to use RL architecture 
    PS_dataset = Image_dataloader(PS_PATH, CSV_PATH, use_all = True, mode  = TRAIN_MODE)    # dataset size = 140
    Data_sampler= DataSampler(PS_dataset)   # dataset size = 140
    Biopsy_env = TemplateGuidedBiopsy_single(Data_sampler, results_dir = LOG_DIR, train_mode = TRAIN_MODE)     # dataset size = 140
    #Biopsy_env = frame_stack_v1(Biopsy_env_init, 3)
    Biopsy_env.reset()

    # Definign RL model to use architecture 
    policy_kwargs = dict(features_extractor_class = NewFeatureExtractor, features_extractor_kwargs=dict(multiple_frames = True, num_channels = 5))
    agent = PPO(CnnPolicy, Biopsy_env, policy_kwargs = policy_kwargs, device = device_cuda, tensorboard_log = LOG_DIR)  # dataset size = 140
    
    if USING_RL:
        print(f"Using RL networks ")
        IL_MODEL = agent.policy.to(device_cuda) # dataset size = 140
        
        if PRETRAIN: 
            MODEL_PATH = PRETRAIN_PATH
            IL_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location = torch.device('cpu')))
        
    else:
        print(f"Not using RL networks ; using feature extractor {FEATURE_EXTRACTOR}")
        
        if FEATURE_EXTRACTOR == 'efficient':
            print(f"Using feature extractor efficient")
            FEATURES_DIM = 512
            INPUT_CHANNELS = 5
            feature_net = EfficientNet3D.from_name("efficientnet-b0",  override_params={'num_classes': FEATURES_DIM}, in_channels=INPUT_CHANNELS)
            #IL_MODEL = ActorCritic_network(feature_net)
            np.where(df_dataset[' num_lesions'] >= 5)[0]
        elif FEATURE_EXTRACTOR == 'vit':
            print(f"Using feature extractor VIT")
            FEATURES_DIM = 512
            INPUT_CHANNELS = 5
            
            feature_net = ViT(
                image_size = 100,          # image size
                frames = 24,               # number of frames
                image_patch_size = 20,     # image patch size # changed to 20 from 16
                frame_patch_size = 2,      # frame patch size
                num_classes = FEATURES_DIM,
                dim = 1024,
                depth = 6,
                heads = 8,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1, 
                channels = INPUT_CHANNELS
            )
            
            #IL_MODEL = ActorCritic_network(feature_net)
            
        else: # use efficientnet 
            print(f"Using feature extractor IMITATIONCONV")
            feature_net = ImitationConv()

        IL_MODEL = ActorCritic_network(feature_net)
    
    # Pre-train using behavioural cloning /imitation learning 

    # Dataset for sub-samples of data : c2l, n2n, l2l, e2e 
    train_ds = Timestep_data_GS(PS_PATH, CSV_PATH, LABELS_PATH, mode = 'train', single_patient = args.single_patient)#.cuda()
    val_ds = Timestep_data_GS(PS_PATH, CSV_PATH, LABELS_PATH, mode = 'val')#.cuda()

    train_dl = DataLoader(train_ds, batch_size = 32, shuffle = True)#.cuda()
    val_dl = DataLoader(val_ds, batch_size = 8, shuffle = False)#.cuda()

    if LOSS_FN == 'MSE':
        print(f"using MSE loss fn")
        all_loss_train, all_loss_val, all_acc_train, all_acc_val  = train_pertimestep_GS(IL_MODEL, agent, train_dl, val_dl, \
        num_epochs = 10000, use_cuda = use_cuda, save_folder = LOG_DIR, clip_actions = clip_actions, device = device_cuda) 
    else:
        print(f"Using HR loss fn")
        all_loss_train, all_loss_val, all_acc_train, all_acc_val  = train_pertimestep_GS_HR(IL_MODEL, agent, train_dl, val_dl, \
        num_epochs = 10000, use_cuda = use_cuda, save_folder = LOG_DIR, clip_actions = clip_actions, device = device_cuda) 

    print('Chicken') 