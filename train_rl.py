#Importing main libraries 
import numpy as np 
import os 
from matplotlib import pyplot as plt
import argparse
from numpy import random
import torch 
from torch.utils.tensorboard import SummaryWriter
from supersuit import frame_stack_v1

#Processes for multi-env processing 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from utils.utils import *
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Tuple 

#Importing module functions 
from utils.Prostate_dataloader import *
#from multipatient_env_v3 import TemplateGuidedBiopsy
#from env.Biopsy_env_single import TemplateGuidedBiopsy_single
from envs.Biopsy_env import TemplateGuidedBiopsy

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

#Stablebaseline functions
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DDPG, SAC, PPO
#from sb3_code.ppo_sb3 import PPO 
from stable_baselines3.ppo.policies import CnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter

#Arg parse functions 
parser = argparse.ArgumentParser(prog='train',
                                description="Train RL agent See list of available arguments for more info.")

parser.add_argument('--log_dir',
                    '--log',
                    metavar='log_dir',
                    type=str,
                    action='store',
                    default='RL_SINGLE',
                    help='Log dir to save results to')

parser.add_argument('--train_multipatient',
                    '--t_multi',
                    metavar='train_multipatient',
                    type=str,
                    action='store',
                    default='True',
                    help='Whether or not to train with multiple patients')

parser.add_argument('--max_num_steps',
                    '--max_steps',
                    metavar='max_num_steps',
                    type=str,
                    action='store',
                    default='20',
                    help='Maximum number of steps before terminating')

parser.add_argument('--index_dataset',
                    '--idx',
                    metavar='index_dataset',
                    type=str,
                    action='store',
                    default='1',
                    help='Index of dataset to train with, if using single patient training')

parser.add_argument('--reward_fn',
                    '--r_fn',
                    metavar='reward_fn',
                    type=str,
                    action='store',
                    default='penalty',
                    help='Which reward fn to use')

parser.add_argument('--num_interactions',
                    '--n_i',
                    metavar='num_interactions',
                    type=str,
                    action='store',
                    default='20',
                    help='Number of timesteps before updating policy (ie training data steps)')

parser.add_argument('--batch_size',
                    '--b_size',
                    metavar='batch_size',
                    type=str,
                    action='store',
                    default='100',
                    help='Size of batch size used for training')

parser.add_argument('--gamma',
                    '--gam',
                    metavar='gamma',
                    type=str,
                    action='store',
                    default='0.99',
                    help='Gamma variable for PPO')

parser.add_argument('--algorithm',
                    '--algorithm',
                    metavar='algorithm',
                    type=str,
                    action='store',
                    default='PPO',
                    help='Which algorithm to use? Default PPO')

parser.add_argument('--gae_lambda',
                    '--gae',
                    metavar='gae_lambda',
                    type=str,
                    action='store',
                    default='0.95',
                    help='Lambda used for GAE variable for PPO')

parser.add_argument('--entropy_coeff',
                    '--ent',
                    metavar='entropy_coeff',
                    type=str,
                    action='store',
                    default='0.0001',
                    help='Entropy coeff used for training')

parser.add_argument('--clip_range',
                    '--clip',
                    metavar='clip_range',
                    type=str,
                    action='store',
                    default='0.2',
                    help='Clip range used for PPO  training')

parser.add_argument('--eval_freq',
                    '--e_freq',
                    metavar='eval_freq',
                    type=str,
                    action='store',
                    default='100',
                    help='How often to evaluate env')

parser.add_argument('--value_coeff',
                    metavar='value_coeff',
                    type=str,
                    action='store',
                    default='0.5',
                    help='Value function coefficient for loss function')

parser.add_argument('--gpu_num',
                    '--gpu',
                    metavar='gpu_num',
                    type=str,
                    action='store',
                    default='1',
                    help='Which GPU to use')

parser.add_argument('--terminating_condition',
                    metavar='terminating_condition',
                    type=str,
                    action='store',
                    default='max_num_steps',
                    help='Which terminating condition to use')

parser.add_argument('--start_centre',
                    metavar='start_centre',
                    type=str,
                    action='store',
                    default='True',
                    help='Whether or not to initialise agent at the centre of dataset')

parser.add_argument('--learning_rate',
                    metavar='learning_rate',
                    type=str,
                    action='store',
                    default='0.0001',
                    help='Learning rate used for training')

parser.add_argument('--pretrain',
                    metavar='pretrain',
                    type=str,
                    action='store',
                    default='True',
                    help='Whether or not to use pre-training with IL weights!')

parser.add_argument('--deform',
                    metavar='deform',
                    type=str,
                    action='store',
                    default='True',
                    help='Whether to deform lesion / prostate glands or not')

parser.add_argument('--deform_rate',
                    metavar='deform_rate',
                    type=str,
                    action='store',
                    default='0.25',
                    help='Rate of control points being used for deformation')

parser.add_argument('--deform_scale',
                    metavar='deform_scale',
                    type=str,
                    action='store',
                    default='0.1',
                    help='Scale of deform poitns being used for deformation')

parser.add_argument('--env',
                    metavar='env',
                    type=str,
                    action='store',
                    default='bug',
                    help='Which environment to use')

parser.add_argument('--pretrain_model',
                    metavar='pretrain_model',
                    type=str,
                    action='store',
                    default='GS',
                    help='Which pre train model to use')

args = parser.parse_args()

def evaluate_agent(agent, num_episodes, Biopsy_env_val):
    """
    A function that periodically updates the agent every now and then
    """
    #Biopsy_env_val = Monitor(Biopsy_env_val, filename= './log')
    
    reward_per_episode = np.zeros(num_episodes)
    all_episode_len = np.zeros(num_episodes)
    #lesions_hit = np.zeros(num_episodes)
    hit_threshold = np.zeros(num_episodes)
    hit_rate = np.zeros(num_episodes)
    ccl_corr_vals = np.zeros(num_episodes)
    efficiency = np.zeros(num_episodes)
    plots = []


    for episode_num in range(num_episodes):
        
        print("\n")
        #Reset environment
        obs = Biopsy_env_val.reset()
        episode_reward = 0
        episode_len = 0 

        done = False

        while not done:
            
            action, _states = agent.predict(obs, deterministic= False)
            #print(f'action : {action}')
            obs, reward, done_info, info = Biopsy_env_val.step(action)
            done = info['new_patient']
            episode_len += 1
            episode_reward += reward

        # Save episode reward 
        reward_per_episode[episode_num] = episode_reward
        all_episode_len[episode_num] = episode_len
        #lesions_hit[episode_num] = int(info['all_lesions_hit'])
        hit_threshold[episode_num] = info['hit_threshold_reached']
        hit_rate[episode_num] = info['hit_rate']
        ccl_corr_vals[episode_num] = info['ccl_corr_online']
        efficiency[episode_num] = info['efficiency']
        #plots.append(info['ccl_plots'])

        print(f"Episode reward : {episode_reward}")
        print(f"Episode_len : {episode_len}")
        print(f"Hit rate : {info['hit_rate']}")
        print(f"Num needles per lesion : {info['num_needles_per_lesion']}")
        print(f"Correlation coeff {info['ccl_corr_online']}")
        print(f"Efficiency {info['efficiency']}")
        
    average_episode_reward = np.nanmean(reward_per_episode)
    std_episode_reward = np.nanstd(reward_per_episode)
    average_episode_len = np.nanmean(all_episode_len)

    #average_lesions_hit = np.nanmean(lesions_hit) * 100
    #average_percentage_hit = np.nanmean(percentage_lesions_hit) * 100

    avg_hit_rate = np.nanmean(hit_rate) * 100 
    avg_ccl_corr = np.nanmean(ccl_corr_vals)
    avg_efficiency = np.nanmean(efficiency) # just hit rate, but / 100
    avg_hit_threshold = np.nanmean(hit_threshold) * 100 # average threshold reached ie 4 needles per lesion achieved

    print(f"Average episode reward {average_episode_reward} +/- {std_episode_reward}")
    print(f"Average episode length {average_episode_len}")
    #print(f"Average percentage of lesions hit {average_percentage_hit}")
    print(f"Average correlation coefficient {avg_ccl_corr}")
    print(f"Average Efficiency {avg_efficiency}")
    
    return average_episode_reward, std_episode_reward, average_episode_len, avg_hit_rate, avg_ccl_corr, avg_efficiency, avg_hit_threshold

if __name__ == '__main__':
    
    # PATHS TO DATASETS
    #DATASET_PATH = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality'
    #RECTUM_PATH = '/Users/ianijirahmae/Documents/PhD_project/rectum_pos.csv'

    DATASET_PATH = '/raid/candi/Iani/MRes_project/Reinforcement Learning/DATASETS/'
    #RECTUM_PATH = '/raid/candi/Iani/MRes_project/Reinforcement Learning/rectum_pos.csv'
    CSV_PATH = '/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv'
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #cuda_1 = torch.device('cuda:1')
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    torch.cuda.is_available() # Check that CUDA works
    print(f"Number GPU available : {torch.cuda.device_count()}")
    print('Using device:', device_cuda)
    print(f' Using env : {args.env}')

    # Obtaining arguments from parsing 
    LOG_DIR = args.log_dir
    os.makedirs(LOG_DIR, exist_ok=True) #Making log directory if not already there 
    train_multipatient = (args.train_multipatient == 'True')
    patient_idx = int(args.index_dataset)
    num_interactions = int(args.num_interactions)
    eval_freq = int(args.eval_freq)
    reward_fn = args.reward_fn
    algorithm = args.algorithm
    env_type = args.env


    #HYPERPARAMETERS 
    BATCH_SIZE = int(args.batch_size)
    MAX_NUM_STEPS = int(args.max_num_steps) #for terminating condition 
    GAMMA = float(args.gamma)
    GAE_LAMBDA = float(args.gae_lambda)
    ENTROPY_COEFF = float(args.entropy_coeff)
    CLIP_RANGE = float(args.clip_range)
    terminating_condition = args.terminating_condition 
    START_CENTRE = (args.start_centre == 'True')
    LEARNING_RATE = float(args.learning_rate)
    PRETRAIN = (args.pretrain == 'True')
    VALUE_COEFF = float(args.value_coeff)
    
    DEFORM = (args.deform == 'True')
    DEFORM_RATE = float(args.deform_rate)
    DEFORM_SCALE = float(args.deform_scale)
    pretrain_model = args.pretrain_model 
    
    print(f"Pretrain : {PRETRAIN}")
    print(f"Deform : {DEFORM}, RATE : {DEFORM_RATE}, SCALE : {DEFORM_SCALE}")
    # 1. Initialise which dataloader to use 
    
    if train_multipatient:
        print("train with multiple patients")
        PS_dataset_train = Image_dataloader(DATASET_PATH, CSV_PATH, use_all = True, mode  = 'train')
        Data_sampler_train = DataSampler(PS_dataset_train)

        # Set up validation dataset too 
        PS_dataset_val = Image_dataloader(DATASET_PATH, CSV_PATH, use_all = True, mode  = 'val')
        Data_sampler_val = DataSampler(PS_dataset_val)

    else:
        print("train with single patients only")
        PS_dataset_train = Image_dataloader_single(DATASET_PATH, CSV_PATH, idx = patient_idx, mode  = 'train')
        Data_sampler_train = DataSampler(PS_dataset_train)

        # Set up validation dataset too 
        PS_dataset_val = Image_dataloader_single(DATASET_PATH, CSV_PATH, idx = patient_idx, mode  = 'val')
        Data_sampler_val = DataSampler(PS_dataset_val)

    # 2. Initialise vectorised environment

    #Vectorise environment - number of envs to use : wrap with monitor env 
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
                if env_type == 'single':
                    Biopsy_env_init = TemplateGuidedBiopsy_single(Data_sampler_train, env_num = rank_num, reward_fn = reward_fn, \
                        max_num_steps = MAX_NUM_STEPS, results_dir= LOG_DIR,  \
                            terminating_condition = terminating_condition, device = device_cuda, \
                                start_centre = START_CENTRE, deform = DEFORM, deform_rate = DEFORM_RATE, deform_scale = DEFORM_SCALE)
                else:
                    print(f'using bug env')
                    Biopsy_env_init = TemplateGuidedBiopsy_bug(Data_sampler_train, env_num = rank_num, reward_fn = reward_fn, \
                        max_num_steps = MAX_NUM_STEPS, results_dir= LOG_DIR,  \
                            terminating_condition = terminating_condition, device = device_cuda, \
                                start_centre = START_CENTRE, deform = DEFORM, deform_rate = DEFORM_RATE, deform_scale = DEFORM_SCALE)
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
    
    Vec_Biopsy_env = make_vec_env(n_envs = 4, vecenv_class = 'Dummy', monitor_dir = LOG_DIR, monitor_kwargs =  {'info_keywords' : ('ccl_corr_online', 'hit_rate', 'efficiency', 'all_lesion_size', 'all_ccl', 'num_needles', 'num_needles_hit')})#.to(device_cuda)
    #Vec_Biopsy_env = VecFrameStack(Vec_Biopsy_env, 3)

    # 3. Initialise training agent 

    policy_kwargs = dict(features_extractor_class = NewFeatureExtractor, features_extractor_kwargs=dict(multiple_frames = True, num_channels = 5))

    if algorithm == 'PPO':
        
        if PRETRAIN:
            
            print(f"PRETRAINING WITH IL AGENT :)")
            
            #MODEL_PATH = 'raid/candi/Iani/Biopsy_RL/agent_model.zip/IL_GS_v2'
            
            if pretrain_model == 'GS':
                print(f"Using gs model for pre-training")
                #MODEL_PATH = 'raid/candi/Iani/Biopsy_RL/agent_model.zip/IL_GS_v2'
                MODEL_PATH = '/raid/candi/Iani/Biopsy_RL/MODELS/GS/best_val_model.pth'
                #MODEL_PATH = '/raid/candi/Iani/Biopsy_RL/IMITATION_SUBSAMPLE_e2e/agent_model.zip'
            elif pretrain_model == 'E2E':
                print("Using E2E model for pre-training")
                #MODEL_PATH = '/raid/candi/Iani/Biopsy_RL/IMITATION_SUBSAMPLE_e2e/agent_model.zip'
                MODEL_PATH = '/raid/candi/Iani/Biopsy_RL/MODELS/GS/best_val_model.pth'
            else:
                print(f"Using wacky model for pre-training")
                MODEL_PATH = 'raid/candi/Iani/Biopsy_RL/agent_model.zip/IL_WACKY_v1'
                
            policy_kwargs = dict(features_extractor_class = NewFeatureExtractor, \
            features_extractor_kwargs=dict(multiple_frames = True, num_channels = 5)) #, activation_fn = torch.nn.Tanh)
            
            if pretrain_model != 'E2E':
                print(f"Not using E2E CUSTOM")
                policy_kwargs = dict(features_extractor_class = NewFeatureExtractor, \
                features_extractor_kwargs=dict(multiple_frames = True, num_channels = 5)) #, activation_fn = torch.nn.Tanh)
                agent = PPO(CnnPolicy, Vec_Biopsy_env, policy_kwargs = policy_kwargs, gamma = GAMMA, vf_coef = VALUE_COEFF, \
                n_steps = num_interactions, gae_lambda = GAE_LAMBDA, batch_size = BATCH_SIZE,\
                n_epochs = 2, learning_rate = LEARNING_RATE, ent_coef = ENTROPY_COEFF, device = device_cuda, tensorboard_log = LOG_DIR)
            else:
                policy_kwargs = dict(features_extractor_class = NewFeatureExtractor, \
                    features_extractor_kwargs=dict(multiple_frames = True, num_channels = 5)) #, activation_fn = torch.nn.Tanh)
                agent = PPO(CustomActorCriticPolicy, Vec_Biopsy_env, policy_kwargs = policy_kwargs, gamma = GAMMA, vf_coef = VALUE_COEFF,\
                n_steps = num_interactions, gae_lambda = GAE_LAMBDA, batch_size = BATCH_SIZE,\
                n_epochs = 2, learning_rate = LEARNING_RATE, ent_coef = ENTROPY_COEFF, device = device_cuda, tensorboard_log = LOG_DIR)
                
            # Loading pre-trained model!! :) 
            # agent = PPO.load(MODEL_PATH, env = Vec_Biopsy_env, gamma = GAMMA, \
            # n_steps = num_interactions, gae_lambda = GAE_LAMBDA, batch_size = BATCH_SIZE,\
            # n_epochs = 2, learning_rate = LEARNING_RATE, ent_coef = ENTROPY_COEFF, device = device_cuda, tensorboard_log = LOG_DIR)
            
            # Loading policy weights into model 
            model = agent.policy 
            if pretrain_model == 'GS':
                #MODEL_IL_PATH = '/raid/candi/Iani/Biopsy_RL/IL_GS_v2/best_val_model.pth'
                MODEL_IL_PATH = '/raid/candi/Iani/Biopsy_RL/MODELS/GS/best_val_model.pth'
            elif pretrain_model == 'E2E':
                print(f"loded IL model weights E2E onto model")
                #MODEL_IL_PATH = '/raid/candi/Iani/Biopsy_RL/IMITATION_SUBSAMPLE_e2e/best_val_model.pth'
                MODEL_IL_PATH = '/raid/candi/Iani/Biopsy_RL/MODELS/GS/best_val_model.pth'
            else:
                #MODEL_IL_PATH = '/raid/candi/Iani/Biopsy_RL/IMITATION_SUBSAMPLE_e2e/best_val_model.pth'
                #MODEL_IL_PATH = '/raid/candi/Iani/Biopsy_RL/IL_WACKY_v1/best_val_model.pth'
                MODEL_IL_PATH = '/raid/candi/Iani/Biopsy_RL/MODELS/IL_WACKY_v1/best_val_model.pth'
            
            model.load_state_dict(torch.load(MODEL_IL_PATH)) #, map_location = torch.device('cpu')))
            
            # Load as agent with loaded weights!!! 
            agent.policy = model 
            #agent.policy = model
             
            print(f'Loaded weights from pretrained IL model {pretrain_model}') 
             
            
        else:
            agent = PPO(CnnPolicy, Vec_Biopsy_env, policy_kwargs = policy_kwargs, gamma = GAMMA, \
            n_steps = num_interactions, gae_lambda = GAE_LAMBDA, batch_size = BATCH_SIZE,\
            n_epochs = 2, learning_rate = LEARNING_RATE, ent_coef = ENTROPY_COEFF, device = device_cuda, tensorboard_log = LOG_DIR)
            
    elif algorithm == 'SAC': 
        agent = SAC(CnnPolicy, Vec_Biopsy_env, policy_kwargs = policy_kwargs, gamma = GAMMA, \
        batch_size = BATCH_SIZE, learning_starts = 500, buffer_size = 10000, train_freq = (10, "episode"), \
        gradient_steps = 2, learning_rate = LEARNING_RATE, device = device_cuda, tensorboard_log = LOG_DIR)
        
    # 4. Train agent, evaluate for number of episodes 
    callback_train = SaveOnBestTrainingReward_single(check_freq=eval_freq, log_dir=LOG_DIR)

    NUM_TRIALS = 1000
    best_reward = -np.inf # set best reward to -inf to save even negative rewards
    best_std = 0 
    
    tensorboard_path = os.path.join(LOG_DIR, 'runs')
    os.makedirs(tensorboard_path, exist_ok=True)

    writer = SummaryWriter(tensorboard_path)

    ## Set up validation sampler, but probably outside loop 
    PS_dataset_val = Image_dataloader(DATASET_PATH, CSV_PATH, mode  = 'val', use_all = True)
    Data_sampler_val = DataSampler(PS_dataset_val)

    if train_multipatient == False:
        print("train with single patients only")

        # Set up validation dataset too 
        PS_dataset_val = Image_dataloader_single(DATASET_PATH, CSV_PATH, idx = patient_idx, mode  = 'val')
        Data_sampler_val = DataSampler(PS_dataset_val)

    for trial_num in range(NUM_TRIALS):

        print(f'\n Trial: {trial_num}')
        print(f"Best val reward : {best_reward} +/- {best_std}")

        print(f'\n Resuming training')
        agent.learn(total_timesteps= 6000, callback = callback_train, tb_log_name = 'PPO', reset_num_timesteps = False)

        print(f'\n Evaluation')
        # Reset validation biopsy env val for each trial, as to compute CCL from scratch again 

        print(f'Using env type : bug for evaluation')
        Biopsy_env_val = TemplateGuidedBiopsy(Data_sampler_val, train_mode = 'val', max_num_steps = MAX_NUM_STEPS, reward_fn = reward_fn, \
            env_num = '1', results_dir= LOG_DIR, device = device_cuda)
            
        #Biopsy_env_val = frame_stack_v1(Biopsy_env_init, 3)
        Biopsy_env_val.reset()

        # Evaluate for 40 episodes at a time using biopsy_env_val
        average_val_reward, std_episode_reward, avg_episode_len, avg_hit_rate, avg_ccl_corr, avg_efficiency, avg_hit_threshold = evaluate_agent(agent, 30, Biopsy_env_val)
        if average_val_reward > best_reward:

            best_reward = copy.deepcopy(average_val_reward)
            best_std = std_episode_reward 

            #Save model if best reward is better than previous reward 
            print(f"Saving new best model")
            best_model_save_path = LOG_DIR
            agent.save(os.path.join(best_model_save_path, "best_model_val"))

        # Save to tensorboard best reward and episode len 
        writer.add_scalar('rollout/episode_reward', average_val_reward, trial_num)
        writer.add_scalar('rollout/episode_len', avg_episode_len, trial_num)
        writer.add_scalar('metrics/avg_hit_rate', avg_hit_rate, trial_num)
        writer.add_scalar('metrics/avg_hit_threshold', avg_hit_threshold, trial_num)
        writer.add_scalar('metrics/efficiency',avg_efficiency, trial_num)
        #writer.add_figure('ccl_plot', ccl_plots[-1], trial_num)
   
    model_path = os.path.join(LOG_DIR, 'final_model') #Name of model to save 
    agent.save(model_path)


print('chicken')
