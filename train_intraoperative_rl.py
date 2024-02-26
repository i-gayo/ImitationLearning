import numpy as np 
import os 
from matplotlib import pyplot as plt
import argparse
from numpy import random
import torch 
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from utils.utils import *
from utils.data_utils import * 
from environment.biopsy_env import * 
from environment.simple_env import TargettingEnv
from utils.Prostate_dataloader import DataSampler_SimpleEnv
from utils.data_utils import BiopsyDataset
from stable_baselines3 import DDPG, SAC, PPO
from stable_baselines3.ppo.policies import CnnPolicy
from networks.intraoperative_rl import Feature_Extractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

parser = argparse.ArgumentParser(prog='train',
                                description="Train RL agent See list of available arguments for more info.")

parser.add_argument('--log_dir',
                    '--log',
                    metavar='log_dir',
                    type=str,
                    action='store',
                    default='Baseline',
                    help='Log dir to save results to')

parser.add_argument('--batch_size',
                    '--b_size',
                    metavar='batch_size',
                    type=str,
                    action='store',
                    default='100',
                    help='Size of batch size used for training')

parser.add_argument('--algorithm',
                    '--algorithm',
                    metavar='algorithm',
                    type=str,
                    action='store',
                    default='PPO',
                    help='Which algorithm to use? Default PPO')

parser.add_argument('--gamma',
                    metavar='gamma',
                    type=str,
                    action='store',
                    default='0.99',
                    help='Gamma variable for PPO')

parser.add_argument('--entropy_coeff',
                    '--ent',
                    metavar='entropy_coeff',
                    type=str,
                    action='store',
                    default='0.0001',
                    help='Entropy coeff used for training')

parser.add_argument('--dataset',
                    metavar='dataset',
                    type=str,
                    action='store',
                    default='baseline',
                    help='Which dataset to use for training with')

parser.add_argument('--learning_rate',
                    metavar='learning_rate',
                    type=str,
                    action='store',
                    default='0.0001',
                    help='Learning rate used for training')
args = parser.parse_args()

if __name__ == '__main__':
    
    # Loading argparse 
    DATASET = args.dataset
    BATCH_SIZE = int(args.batch_size) 
    GAMMA = float(args.gamma)
    LEARNING_RATE = float(args.learning_rate)
    NUM_TRIALS = 1000
    EVAL_FREQ = 100 # how many times to check model for (about 5 episodes)
    BASE_DIR = 'RL_models'
    LOG_DIR = os.path.join(BASE_DIR, args.log_dir)
    os.makedirs(LOG_DIR, exist_ok = True)
    MODEL_DIR = os.path.join(LOG_DIR, 'models')
    os.makedirs(MODEL_DIR, exist_ok = True)
    
    # LOADING DEVICE: 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load dataset folder for each one 
    if DATASET != 'Diffusion':
        DS_PATH = '/raid/candi/Iani/mr2us/ALL_DATA/RL_pix2pix'
    elif DATASET == 'transformnet':
        raise NotImplementedError("Not yet implemented for transfromnet")
    else:
        DS_PATH = '/raid/candi/Iani/mr2us/ALL_DATA/RL_diffusion'
    
    # Set give_fake is fake for baseline experiments 
    if DATASET == 'baseline':
        give_fake = False
    else:
        give_fake = True 

    ########## LOAD DATASET and initialise VEC ENV ##########
    
    ds = BiopsyDataset(DS_PATH, mode = 'test', give_fake = give_fake)    
    #dl = DataLoader(ds, batch_size = 1)
    data_sampler = DataSampler_SimpleEnv(ds)
    biopsy_env = TargettingEnv(data_sampler, device= device)
    
    def make_vec_env(n_envs, vecenv_class = 'Dummy', monitor_dir = './log', monitor_kwargs = None, seed=0):
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environments you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """

        def make_env(rank, seed = 0):
            def _init():
                
                rank_num = str(rank)
                #Initialise environment
                TargettingEnv(data_sampler)
                Biopsy_env_init = TargettingEnv(data_sampler, device = device)
                #Biopsy_env_init = frame_stack_v1(Biopsy_env_init, 3)
                Biopsy_env_init.reset()
                #Biopsy_env_init.seed(seed + rank)
                Biopsy_env_init.action_space.seed(seed + rank)
            
                # Wrap the env in a Monitor wrapper
                # to have additional training information
                monitor_path = os.path.join(monitor_dir, str(rank)) 
                # Create the monitor folder if needed
                if monitor_path is not None:
                    os.makedirs(monitor_dir, exist_ok=True)

                env = Monitor(Biopsy_env_init, filename=monitor_path, **monitor_kwargs)

                return env

            set_random_seed(seed)
            return _init        

        if vecenv_class == 'Dummy':
            return DummyVecEnv([make_env(i) for i in range(n_envs)])
        else: 
            return SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    Vec_Biopsy_env = make_vec_env(n_envs = 4, vecenv_class = 'Dummy', monitor_dir = LOG_DIR, monitor_kwargs =  {'info_keywords' : ('ccl',)})#.to(device_cuda)
    
    ########## LOAD MODELS ##########
    
    NUM_INTERACTIONS = 100  # how many steps before updating each step 
    policy_kwargs = dict(features_extractor_class = Feature_Extractor)
    
    agent = PPO(CnnPolicy, 
                Vec_Biopsy_env, 
                policy_kwargs = policy_kwargs, 
                gamma = GAMMA, 
                n_steps = NUM_INTERACTIONS, # 100 = 5 episodes before updating?
                batch_size = BATCH_SIZE,
                n_epochs = 2, 
                learning_rate = LEARNING_RATE,
                device = device, 
                tensorboard_log = LOG_DIR)
    
    ####### TRAIN RANDOM POLICY #######
    callback_train = SaveOnBestTrainingReward_single(check_freq=EVAL_FREQ, log_dir=LOG_DIR)
    best_reward = -np.inf # set best reward to -inf to save even negative rewards
    best_std = 0 
    
    tensorboard_path = os.path.join(LOG_DIR, 'runs')
    os.makedirs(tensorboard_path, exist_ok=True)

    writer = SummaryWriter(tensorboard_path)

    ## Initialise training procedure:
    
    for trial_num in range(NUM_TRIALS):
        # Learn for 100 episodes (20 x 100), then repeat 
        print(f'\n Trial: {trial_num} starting training')
        agent.learn(total_timesteps= 2000, callback = callback_train, tb_log_name = 'PPO', reset_num_timesteps = False)

        model_path = os.path.join(MODEL_DIR, f'model_{trial_num}') #Name of model to save 
        agent.save(model_path)
    
    print("Finished training")
    
    ########## EVALUATE RANDOM POLICY ##########
    # NUM_EPISODES = 10
    
    # done = False 
    # for i in range(NUM_EPISODES):
        
    #     obs = biopsy_env.reset()
        
    #     while not done: 
    #         action,_ = agent.predict(obs.squeeze())
    #         obs, reward, terminated, truncated, info = biopsy_env.step()
            
    #         # fig, axs = plt.subplots(1,2)
    #         # axs[0].imshow(obs[-1,:,:])
    #         # axs[1].imshow(obs[-2,:,:])
    #         # plt.savefig("IMGS/NEW_OBS.png")
            
    #         print(f'Reward : {reward}')
            
    #     print('fuecoco')
            
        
    ##### GENERATING VECTORISED ENVS 
