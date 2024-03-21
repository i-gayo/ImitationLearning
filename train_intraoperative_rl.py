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
import copy 

parser = argparse.ArgumentParser(prog='train',
                                description="Train RL agent See list of available arguments for more info.")

parser.add_argument('--log_dir',
                    '--log',
                    metavar='log_dir',
                    type=str,
                    action='store',
                    default='Baseline_CHECK',
                    help='Log dir to save results to')

parser.add_argument('--load_checkpoint',
                    metavar='load_checkpoint',
                    type=str,
                    action='store',
                    default='False',
                    help='Whether to resume training from previous checkpoint or to start from scratch')

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

parser.add_argument('--reward_fn',
                    metavar='reward_fn',
                    type=str,
                    action='store',
                    default='obs',
                    help='Reward function used for training')

parser.add_argument('--change_voxdims',
                    metavar='change_voxdims',
                    type=str,
                    action='store',
                    default='True',
                    help='Whether to change voxdims')

parser.add_argument('--max_num_steps',
                    metavar='max_num_steps',
                    type=str,
                    action='store',
                    default='20',
                    help='How many steps to do before ending episode!!!')

parser.add_argument('--inc_shaped',
                    metavar='inc_shaped',
                    type=str,
                    action='store',
                    default='False',
                    help='How many steps to do before ending episode!!!')

parser.add_argument('--use_all',
                    metavar='--use_all',
                    type=str,
                    action='store',
                    default='True',
                    help='Whether to use all patients for training or just some!')

parser.add_argument('--num_train',
                    metavar='--num_train',
                    type=str,
                    action='store',
                    default='5',
                    help='How many patients to train with')

args = parser.parse_args()

if __name__ == '__main__':
    
    # Loading argparse 
    DATASET = args.dataset
    BATCH_SIZE = int(args.batch_size) 
    GAMMA = float(args.gamma)
    LEARNING_RATE = float(args.learning_rate)
    NUM_TRIALS = 10000
    EVAL_FREQ = 100 # how many times to check model for (about 5 episodes)
    MAX_STEPS = int(args.max_num_steps) 
    LOAD_CHECKPOINT = (args.load_checkpoint == 'True')
    
    BASE_DIR = 'RL_models'
    LOG_DIR = os.path.join(BASE_DIR, args.log_dir)
    os.makedirs(LOG_DIR, exist_ok = True)
    MODEL_DIR = os.path.join(LOG_DIR, 'models')
    os.makedirs(MODEL_DIR, exist_ok = True)
    CHANGE_VOXDIMS = (args.change_voxdims == 'True')
    print(f"CHanging voxdims {CHANGE_VOXDIMS}")
    REWARD_FN = args.reward_fn
    INC_SHAPED = (args.inc_shaped == 'True')
    USE_ALL = (args.use_all == 'True')
    NUM_TRAIN = int(args.num_train)
    
    print(f"Max num steps : {MAX_STEPS} reward function {REWARD_FN} inc shaped : {INC_SHAPED}")

    # LOADING DEVICE: 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load dataset folder for each one 
    if (DATASET == 'baseline'):
        DS_PATH = '/raid/candi/Iani/mr2us/ALL_DATA/RL_pix2pix'
        checkpoint_path = './RL_models/NEW_HIT_BASELINE_3/models/model_36.zip'
        best_mean = -8
    elif (DATASET == 'pix2pix'):
        DS_PATH = '/raid/candi/Iani/mr2us/ALL_DATA/RL_pix2pix'
        checkpoint_path = './RL_models/NEW_HIT_PIX2PIX_3/models/model_229.zip'
        best_mean = 5
    elif DATASET == 'transformnet':
        DS_PATH = '/raid/candi/Iani/mr2us/ALL_DATA/RL_NEWTRANSFORM'
        checkpoint_path = './RL_models/NEW_HIT_TRANSFORM_4/models/model_11.zip'
        best_mean = -11
    else:
        DS_PATH = '/raid/candi/Iani/mr2us/ALL_DATA/RL_NEWDIFFUSION'
        checkpoint_path = './RL_models/NEW_HIT_DIFFUSION_3/models/model_126.zip'
        best_mean = -3
    
    # Set give_fake is fake for baseline experiments 
    if DATASET == 'baseline':
        give_fake = False
    else:
        give_fake = True 

    ########## LOAD DATASET and initialise VEC ENV ##########
    
    ds = BiopsyDataset(DS_PATH, mode = 'test', give_fake = give_fake, sub_mode= 'train', use_all = USE_ALL, num_train = NUM_TRAIN)    
    #dl = DataLoader(ds, batch_size = 1)
    data_sampler = DataSampler_SimpleEnv(ds)
    biopsy_env = TargettingEnv(data_sampler, device= device)
    print(f"Using dataset : using all : {USE_ALL} num_train : {NUM_TRAIN}")
    
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
                Biopsy_env_init = TargettingEnv(data_sampler, 
                                                device = device, 
                                                change_voxdims = CHANGE_VOXDIMS, 
                                                reward_fn = REWARD_FN,
                                                max_steps = MAX_STEPS, 
                                                inc_shaped = INC_SHAPED)
                
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
    
    Vec_Biopsy_env = make_vec_env(n_envs = 4, vecenv_class = 'Dummy', monitor_dir = LOG_DIR, monitor_kwargs =  {'info_keywords' : ('ccl','ccl_v2', 'ccl_v3', 'hr',)})#.to(device_cuda)
    
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
    
    if LOAD_CHECKPOINT:
        agent = PPO.load(checkpoint_path, 
                        Vec_Biopsy_env, 
                        device = device, 
                        policy_kwargs = policy_kwargs, 
                        gamma = GAMMA, 
                        n_steps = NUM_INTERACTIONS, # 100 = 5 episodes before updating?
                        batch_size = BATCH_SIZE,
                        n_epochs = 2, 
                        learning_rate = LEARNING_RATE,
                        tensorboard_log = LOG_DIR)
        
        print(f"Initialising model from previous checkpoint : {checkpoint_path} with best mean {best_mean}")
        
    ####### TRAIN RANDOM POLICY #######
    callback_train = SaveOnBestTrainingReward_single(check_freq=EVAL_FREQ, log_dir=LOG_DIR, best_mean_reward = best_mean)
    best_reward = -np.inf # set best reward to -inf to save even negative rewards
    best_std = 0 
    
    tensorboard_path = os.path.join(LOG_DIR, 'runs')
    os.makedirs(tensorboard_path, exist_ok=True)
    writer = SummaryWriter(tensorboard_path)

    ## Initialise training procedure:
    
    def evaluate_agent(agent, num_episodes, Biopsy_env_val):
        """
        A function that periodically updates the agent every now and then
        """
        #Biopsy_env_val = Monitor(Biopsy_env_val, filename= './log')
        
        reward_per_episode = np.zeros(num_episodes)
        all_episode_len = np.zeros(num_episodes)
        #lesions_hit = np.zeros(num_episodes)
        ccl = np.zeros(num_episodes)
        hit_rate = np.zeros(num_episodes)

        for episode_num in range(num_episodes):
            print("\n")
            #Reset environment
            obs = Biopsy_env_val.reset()
            episode_reward = 0
            episode_len = 0 

            done = False

            while not done:
                
                action, _states = agent.predict(obs.squeeze(), deterministic= True)
                #print(f'action : {action}')
                obs, reward, done, info = Biopsy_env_val.step(action)
                #done = info['new_patient']
                episode_len += 1 
                episode_reward += reward

            # Save episode reward 
            reward_per_episode[episode_num] = episode_reward
            all_episode_len[episode_num] = episode_len
            #lesions_hit[episode_num] = int(info['all_lesions_hit'])
            ccl[episode_num] = info['ccl_v3']
            hit_rate[episode_num] = info['hr']

            #plots.append(info['ccl_plots'])

            print(f"Episode reward : {episode_reward}")
            print(f"Episode_len : {episode_len}")
            print(f"Hit rate : {info['hr']}")
            print(f"CCL {info['ccl_v3']}")

            
        average_episode_reward = np.nanmean(reward_per_episode)
        std_episode_reward = np.nanstd(reward_per_episode)
        average_episode_len = np.nanmean(all_episode_len)

        #average_lesions_hit = np.nanmean(lesions_hit) * 100
        #average_percentage_hit = np.nanmean(percentage_lesions_hit) * 100

        avg_hit_rate = np.nanmean(hit_rate) * 100 
        avg_ccl = np.nanmean(ccl)

        print(f"Average episode reward {average_episode_reward} +/- {std_episode_reward}")
        print(f"Average episode length {average_episode_len}")
        #print(f"Average percentage of lesions hit {average_percentage_hit}")
        print(f"Average HR {avg_hit_rate}")
        print(f"Average CCL {avg_ccl}")
        
        return average_episode_reward, std_episode_reward, average_episode_len, avg_hit_rate, avg_ccl

    # Validation dataset! 
    val_ds = BiopsyDataset(DS_PATH, mode = 'test', give_fake = give_fake, sub_mode= 'test')    
    val_sampler = DataSampler_SimpleEnv(val_ds)
    val_biopsy_env = TargettingEnv(val_sampler, device= device)
    best_val_reward = -np.inf
    
    for trial_num in range(NUM_TRIALS):
        # Learn for 100 episodes (20 x 100), then repeat 
        
        print(f'\n Trial: {trial_num} starting training')
        agent.learn(total_timesteps= 2000, callback = callback_train, tb_log_name = 'PPO', reset_num_timesteps = False)

        model_path = os.path.join(MODEL_DIR, f'model_{trial_num}') #Name of model to save 
        agent.save(model_path)
        
        # Evaluate model on evaluation dataset
        print(f"Validation on validation set")
        average_episode_reward, std_episode_reward, avg_episode_len, avg_hit_rate, avg_ccl = evaluate_agent(agent, num_episodes = len(val_ds), Biopsy_env_val = val_biopsy_env)
        
        if average_episode_reward > best_val_reward:
            best_reward = copy.deepcopy(average_episode_reward)
            print(f"Saving new best model best reward : {best_reward}")
            best_model_save_path = LOG_DIR
            val_model_path = os.path.join(MODEL_DIR, "VALIDATION_MODEL")
            agent.save(val_model_path)
            
        writer.add_scalar('rollout_val/episode_reward', average_episode_reward, trial_num)
        writer.add_scalar('rollout_val/episode_len', avg_episode_len, trial_num)
        writer.add_scalar('metrics_val/avg_hit_rate', avg_hit_rate, trial_num)
        writer.add_scalar('metrics_val/ccl',avg_ccl, trial_num)
        
    
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
