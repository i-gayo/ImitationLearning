#Importing main libraries 
import numpy as np 
import os 
from matplotlib import pyplot as plt
import argparse
from numpy import random
import torch 
from utils.Prostate_dataloader import * 
from utils.rl_utils import Image_dataloader
from utils_il import * 
import pandas as pd 
#matplotlib.use('TkAgg')
import gym
import numpy as np
from scipy.spatial.distance import cdist as cdist  

#Importing module functions 
from utils.Prostate_dataloader import *
from envs.Biopsy_env import TemplateGuidedBiopsy_single
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import CnnPolicy
import torch.nn.functional as F
import nibabel as nib 
from stable_baselines3.common.monitor import Monitor

#Arg parse functions 
parser = argparse.ArgumentParser(prog='testn',
                                description="Testing RL agents.")

parser.add_argument('--mode',
                    metavar='mode',
                    type=str,
                    action='store',
                    default='RL',
                    help='Which method of training : RL or IL')

parser.add_argument('--file_name',
                    metavar='file_name',
                    type=str,
                    action='store',
                    default='results',
                    help='Name of csv path to save to')

parser.add_argument('--dataset',
                    metavar='dataset',
                    type=str,
                    action='store',
                    default='train',
                    help='training, or testing or validation')

parser.add_argument('--deform',
                    metavar='deform',
                    type=str,
                    action='store',
                    default='False',
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

parser.add_argument('--tre',
                    metavar='tre',
                    type=str,
                    action='store',
                    default='3',
                    help='TRE used for reg experiments')

parser.add_argument('--use_rl',
                    metavar='use_rl',
                    type=str,
                    action='store',
                    default='False',
                    help='Whether to use RL outputs or IL outputs')



def upsample_3d_tensor(input_tensor, target_dimensions):
    """
    Upsample a 3D PyTorch tensor to the given target dimensions.

    Parameters:
    - input_tensor (torch.Tensor): The input tensor to upsample.
    - target_dimensions (tuple): The target dimensions (depth, height, width).

    Returns:
    - torch.Tensor: The upsampled tensor.
    """
        # Add batch and channel dimensions for 3D tensor
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    
    # Calculate the scale factors for each dimension
    scale_factors = [
        target_dim / input_dim
        for target_dim, input_dim in zip(target_dimensions, input_tensor.shape)
    ]

    # Perform upsampling using trilinear interpolation for 3D tensors
    upsampled_tensor = F.interpolate(
        input_tensor,
        size=target_dimensions,
        mode="trilinear",
        align_corners=False,
    )

    return upsampled_tensor.squeeze()

def torch_tensor_to_nifti(tensor, filename):
    # Convert PyTorch tensor to NumPy array
    data_array = tensor.cpu().numpy()

    # Create a NIfTI image from the NumPy array
    nifti_image = nib.Nifti1Image(data_array, affine=np.eye(4))

    # Save the NIfTI image to a file
    nib.save(nifti_image, filename)

def compute_coverage(fired_pos, lesion_hit, use_all = False):
    """
    A function that computes coverage as std(x)*std(y) * pi of all fired needle positions

    """
    
    # only compute coverage of needles that actually hit lesion 
    lesion_hit = np.array(lesion_hit)
    
    if use_all: 
        fired_x = fired_pos[:,0]
        fired_y = fired_pos[:,1]
    else: 
        fired_x = fired_pos[lesion_hit == True,0]
        fired_y = fired_pos[lesion_hit == True,1]
    
    std_x = np.std(fired_x)
    std_y = np.std(fired_y)
    
    if (std_x == 0) and (std_y != 0):
        std_x = 1
    
    if (std_y == 0) and (std_x != 0):
        std_y = 1
    
    fired_area = (std_x * std_y * np.pi) # area of ellipse 
    
    return fired_area 

def compute_norm_coverage(fired_pos, lesion_hit, lesion_obs):
    """
    A function that computes coverage as std(x)*std(y) * pi of all fired needle positions

    """
    
    # only compute coverage of needles that actually hit lesion 
    lesion_hit = np.array(lesion_hit)
    
    fired_x = fired_pos[lesion_hit == True,0]
    fired_y = fired_pos[lesion_hit == True,1]
    
    std_x = np.std(fired_x)
    std_y = np.std(fired_y)
    
    if (std_x == 0) and (std_y != 0):
        std_x = 1
    
    if (std_y == 0) and (std_x != 0):
        std_y = 1
    
    fired_area = (std_x * std_y * np.pi) # area of ellipse 
    
    # Compute max area
    lesion_obs =  np.max(lesion_obs[0,:,:,:,].numpy(), axis = 2) # maximum projeciton 
    lesion_area = np.sum(lesion_obs) # count number of non-zero pixels as apporximate to 2D area 

    norm_area = fired_area / lesion_area 
    
    return norm_area

def evaluate_agent(agent, num_episodes, Biopsy_env_val, evaluate_mode = 'RL', file_name = 'results', use_rl = False, log_dir = 'log_dir'):
    """
    A function that periodically updates the agent every now and then
    """
    #Biopsy_env_val = Monitor(Biopsy_env_val, filename= './log')
    
    # Create lesion and prostate masks files 
    lesion_folder = os.path.join(log_dir, 'lesion_masks')
    prostate_folder = os.path.join(log_dir, 'prostate_masks')
    os.makedirs(lesion_folder, exist_ok = True)
    os.makedirs(prostate_folder, exist_ok = True)
    print(f"lesion folder created : {lesion_folder} and {prostate_folder}")
    
    reward_per_episode = np.zeros(num_episodes)
    all_episode_len = np.zeros(num_episodes)
    #lesions_hit = np.zeros(num_episodes)
    hit_threshold = np.zeros(num_episodes)
    hit_rate = np.zeros(num_episodes)
    ccl_corr_vals = np.zeros(num_episodes)
    efficiency = np.zeros(num_episodes)
    plots = []
    all_ccl = []
    all_sizes = [] 
    all_norm_ccl = [] 
    all_coverage = [] 
    all_area = [] 
    all_norm_coverage = [] 

    for episode_num in range(num_episodes):
        # For first one, make pandas dataframe
        
        print("\n")
        #Reset environment
        obs = Biopsy_env_val.reset()

        episode_reward = 0
        episode_len = 0 
    
        done = False
        
        fired_pos = [] 
        lesion_hit = [] 
        episode_norm_ccl = []
        episode_ccl = [] 
        
        while not done:
            
            action_rl, _states = agent.predict(obs, deterministic= False)
            model = agent.policy
            
            if evaluate_mode == 'RL':
                obs = obs.unsqueeze(0)
                obs = obs.to(torch.device('cuda'))
                    
            action, value, log_prob = model(obs)
            action = torch.clamp(action, -1, 1)

            #print(f"Action rl : {action_rl} action : {action}")
            if use_rl:
                given_actions = action_rl
    
            else: 
                given_actions = copy.deepcopy(action.detach())[0]
                
            print(f"given actiosn {given_actions} vs rl {action_rl}")
            #print(f'action : {action}')
            obs, reward, done_info, info = Biopsy_env_val.step(given_actions)
            lesion_centre = info['lesion_centre']
            done = info['new_patient']
            print(f"Reward : {reward} Value : {value} \n")
            
            ### Save lesion and prostate deformed masks
            
            # lesion masks 
            patient_name = info['patient_name'][0]
            lesion_idx = info['lesion_idx']
            lesion_def = upsample_3d_tensor(obs[0,:,:,:], (200,200,96))
            img_name  = str(lesion_idx+1) + '_' + str(patient_name) #2_Patient400222898_study_0.nii.gz
            path = os.path.join(lesion_folder, img_name)
            torch_tensor_to_nifti(lesion_def, path)
            
            # Save prostate masks
            prostate_def = upsample_3d_tensor(obs[1,:,:,:], (200,200,96))
            prostate_path = os.path.join(prostate_folder, patient_name)
            torch_tensor_to_nifti(prostate_def, prostate_path)
            
            episode_len += 1
            episode_reward += reward
            
            ccl_val = info['ccl']
            norm_ccl = info['norm_ccl']
            all_norm_ccl.append(norm_ccl)
            current_pos = info['current_pos']
            
            if given_actions[-1] >= -0.33: # ie fired position
                episode_norm_ccl.append(norm_ccl)
                episode_ccl.append(ccl_val)
                fired_pos.append(current_pos[:-1])
                lesion_hit.append(info['needle_hit'])
            
        # Compute metrics for saving calculations 
        dist_all = compute_dist_to_centre(lesion_centre, fired_pos)#, all_hit)
        dist_hit = compute_dist_to_centre(lesion_centre, fired_pos, lesion_hit) 
        range_x, range_y = compute_range(fired_pos, lesion_hit, use_all = True)
        range_x_hit, range_y_hit = compute_range(fired_pos, lesion_hit, use_all = False)
        
        lesion_size = info['lesion_size']
        lesion_idx = info['lesion_idx']
        patient_name = info['patient_name']
        

        # AREA METRICS (spread of needles)
        # Compute coverage 
        
        if len(fired_pos) != 0: 
            all_fired_pos = np.stack(fired_pos)
            coverage = compute_coverage(all_fired_pos, lesion_hit, use_all = True)
        else: 
            # no needles fired 
            coverage = 0
        all_coverage.append(coverage)
        
        # Compute max area
        lesion_obs =  np.max(obs[0,:,:,:,].numpy(), axis = 2) # maximum projeciton 
        lesion_area = np.sum(lesion_obs) # count number of non-zero pixels as apporximate to 2D area 
        all_area.append(lesion_area)
    
        # normalised area 
        all_norm_coverage.append(coverage / lesion_area)
        
        # debugging plots:
        firing_grid = info['firing_grid']
        lesion_img = np.max(obs[0,:,:,:].numpy(), axis = 2)
        lesion_mask_all = np.max(info['lesion_mask'], axis = 2)

        # Save episode reward 
        reward_per_episode[episode_num] = episode_reward
        all_episode_len[episode_num] = episode_len
        #lesions_hit[episode_num] = int(info['all_lesions_hit'])
        hit_threshold[episode_num] = info['hit_threshold_reached']
        hit_rate[episode_num] = info['hit_rate']
        ccl_corr_vals[episode_num] = info['ccl_corr_online']
        efficiency[episode_num] = info['efficiency']
        #plots.append(info['ccl_plots'])
        all_ccl.append(info['all_ccl'])
        all_sizes.append(info['all_lesion_size'])
        
        # Save to pandas data frame 
        ccl_norm_hit = np.mean(np.array(episode_norm_ccl)[lesion_hit])
        ccl_norm_all = np.mean(np.array(episode_norm_ccl))
        ccl_all = np.mean(np.array(episode_ccl))
        ccl_hit = np.mean(np.array(episode_ccl)[lesion_hit])
        
        if episode_num == 0: 
            data = {'Patient' : patient_name, 'lesion': lesion_idx, 'lesion_size' : lesion_size, \
                    'HR' : hit_rate[episode_num], 'norm_ccl_all': ccl_norm_all, 'norm_ccl_hit' : ccl_norm_hit, 'ccl_all' : ccl_all, 'ccl_hit' : ccl_hit, \
                        'norm_coverage' : all_norm_coverage[episode_num],\
                        'ccl_coeff' : ccl_corr_vals[episode_num], 'episode_reward' :reward_per_episode[episode_num], 'episode_len' : episode_len,\
                            'dist_all' : dist_all, 'dist_hit': dist_hit, 'range_x' : range_x, 'range_y' : range_y, 'range_x_hit' : range_x_hit, 'range_y_hit' : range_y_hit}
            df = pd.DataFrame(data, index = [0])
        
        else: 
            new_data = {'Patient' : patient_name, 'lesion': lesion_idx, 'lesion_size' : lesion_size, \
                    'HR' : hit_rate[episode_num], 'norm_ccl_all': ccl_norm_all, 'norm_ccl_hit' : ccl_norm_hit, 'ccl_all' : ccl_all, 'ccl_hit' : ccl_hit, \
                        'norm_coverage' : all_norm_coverage[episode_num],\
                        'ccl_coeff' : ccl_corr_vals[episode_num], 'episode_reward' :reward_per_episode[episode_num], 'episode_len' : episode_len,\
                            'dist_all' : dist_all, 'dist_hit': dist_hit, 'range_x' : range_x, 'range_y' : range_y, 'range_x_hit' : range_x_hit, 'range_y_hit' : range_y_hit}
            
            #df.append(new_data, ignore_index = True)
            new_df = pd.DataFrame(new_data, index = [0])
            
            df = pd.concat([df, new_df], ignore_index = True)
            

        print(f"Episode reward : {episode_reward}")
        print(f"Episode_len : {episode_len}")
        print(f"Hit rate : {info['hit_rate']}")
        print(f"Num needles per lesion : {info['num_needles_per_lesion']}")
        print(f"Correlation coeff {info['ccl_corr_online']}")
        print(f"Efficiency {info['efficiency']}")
    

    result_name = file_name + '.csv'
    result_path = os.path.join(log_dir, result_name)
    print(f"Saving to result path {result_path}")
    df.to_csv(result_path)
    average_episode_reward = np.nanmean(reward_per_episode)
    std_episode_reward = np.nanstd(reward_per_episode)
    average_episode_len = np.nanmean(all_episode_len)

    avg_hit_rate = np.nanmean(hit_rate) * 100 
    std_hit_rate = np.nanstd(hit_rate)
    avg_ccl_corr = np.nanmean(ccl_corr_vals)
    std_corr = np.nanstd(ccl_corr_vals)
    avg_efficiency = np.nanmean(efficiency) # just hit rate, but / 100
    avg_hit_threshold = np.nanmean(hit_threshold) * 100 # average threshold reached ie 4 needles per lesion achieved
    
    # Print coverage metrics: 
    avg_norm_ccl = np.nanmean(all_norm_ccl) 
    std_norm_ccl = np.nanstd(all_norm_ccl)
    avg_ccl = np.nanmean(all_ccl)
    std_ccl = np.nanmean(all_ccl)
    all_norm_coverage = np.array(all_norm_coverage)
    avg_norm_coverage = np.nanmean(all_norm_coverage[~np.isinf(all_norm_coverage)])
    std_norm_coverage = np.nanstd(all_norm_coverage[~np.isinf(all_norm_coverage)])

    print(f"Average episode reward {average_episode_reward} +/- {std_episode_reward}")
    print(f"Average episode length {average_episode_len}")
    #print(f"Average percentage of lesions hit {average_percentage_hit}")
    print(f"Average correlation coefficient {avg_ccl_corr}")
    print(f"Average Efficiency {avg_efficiency} +- {std_hit_rate}")
    print(f"Aveage norm ccl : {avg_norm_ccl} +- {std_norm_ccl}")
    print(f"Average norm coverage : {avg_norm_coverage} +- {std_norm_coverage}")
    print(f"Average CCL : {avg_ccl} +- {std_ccl}")
    
    return average_episode_reward, std_episode_reward, average_episode_len, avg_hit_rate, avg_ccl_corr, avg_efficiency, avg_hit_threshold, all_ccl, all_sizes

def compute_dist_to_centre(lesion_centre, fired_pos, all_hit = None):
    """
    Computes dist to lesion centre 
    
    lesion_centre: ndarray
    fired_pos : (list) of fired postions 
    all_hit : (bool list) bool list of whether needle pos was hit or not; if given, compute mean dist of points that hit lesion. Otherwise, compute all fired pos dist
    """
    
    # Stack fired pos
    grid_pos = np.stack(copy.deepcopy(fired_pos))
    
    if not(all_hit == None):
        mean_dist = np.mean(np.sqrt(np.sum((grid_pos[all_hit] - lesion_centre[:-1])**2, axis = 1)))
    else:
        mean_dist = np.mean(np.sqrt(np.sum((grid_pos - lesion_centre[:-1])**2, axis = 1)))
    
    return mean_dist 

def compute_range(fired_pos, lesion_hit, use_all = False):
    """
    Compute range of x and y values of fired positions
    """
    
    # only compute coverage of needles that actually hit lesion 
    lesion_hit = np.array(lesion_hit)
    fired_pos = np.stack(fired_pos)
    
    if use_all: 
        fired_x = fired_pos[:,0]
        fired_y = fired_pos[:,1]
    else: 
        fired_x = fired_pos[lesion_hit == True,0]
        fired_y = fired_pos[lesion_hit == True,1]
    
    # if nonzero, 
    if len(fired_x) <= 1:
        range_x = 0
    else: 
        range_x = np.ptp(fired_x)
    
    if len(fired_y) <= 1:
        range_y = 0 
    else:
        range_y = np.ptp(fired_y)
    
    # if (std_x == 0) and (std_y != 0):
    #     std_x = 1
    
    # if (std_y == 0) and (std_x != 0):
    #     std_y = 1
    
    # fired_area = (std_x * std_y * np.pi) # area of ellipse 
    
    return range_x, range_y 

if __name__ == '__main__':

    ###### Statistics to obtain: ######
    # HR 
    # average CCL (out of maximum CCL possible) 
    # CCL vs lesion size coefficient (across different lesions)
    # Coverage (how spread are the needles for different strategies)

    # TODO: Change DATASET PATHS     
    DATASET_PATH = '/raid/candi/Iani/MRes_project/Reinforcement Learning/DATASETS/'
    CSV_PATH = '/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv'
    args = parser.parse_args()
    
    # For applying deformation scales 
    DEFORM = (args.deform == 'True')
    DEFORM_RATE = float(args.deform_rate)
    DEFORM_SCALE = float(args.deform_scale)
    USE_RL = (args.use_rl == 'True')
    TRE = float(args.tre)
    
    print(f"Deform : {DEFORM} rate : {DEFORM_RATE} scale {DEFORM_SCALE}")
    
    # TODO : Load your LOG dir here 
    LOG_DIR = '' # This is folder where your models are saved eg : LOG_DIR = '/raid/candi/Iani/Biopsy_RL/D025_025' but save to the folder where your models are saved! 
    RESULTS_DIR = os.path.join(LOG_DIR, "evaluate_ipcai") #this is where your lesion masks, prostate masks and 3 csv files will be saved
    os.makedirs(RESULTS_DIR, exist_ok=True) 
    plt.close('all')
    
    # PARAMETERS 
    MAX_NUM_STEPS = 20
    MODE = args.dataset
    evaluate_mode = args.mode 
    
    print(f"Using model : {args.model} type : {evaluate_mode} using dataset : {MODE}")

    if evaluate_mode == 'IL':
        MODEL_PATH = os.path.join(LOG_DIR, "best_val_model.pth")
    else:
        MODEL_PATH = os.path.join(LOG_DIR,"best_model_val")
    
    # Dataloader 
    PS_dataset = Image_dataloader(DATASET_PATH, CSV_PATH, use_all = True, mode  = MODE)    
    Data_sampler = DataSampler(PS_dataset)

    # Initialising environment  
    Biopsy_env= TemplateGuidedBiopsy_single(Data_sampler, reward_fn = 'penalty', terminating_condition = 'more_than_5', \
    start_centre = True, train_mode = MODE, env_num = '1', max_num_steps = MAX_NUM_STEPS, results_dir= RESULTS_DIR, \
        deform = DEFORM, deform_rate = DEFORM_RATE, deform_scale = DEFORM_SCALE, tre = TRE)
    Biopsy_env = Monitor(Biopsy_env, filename= './log')
    
    print(f"Using tre : {TRE}")
    # Loading models 
    #model = PPO.load(MODEL_PATH, Biopsy_env)
    
    if evaluate_mode == 'IL':

        policy_kwargs = dict(features_extractor_class = NewFeatureExtractor, features_extractor_kwargs=dict(multiple_frames = True, num_channels = 5))
        agent = PPO(CnnPolicy, env = Biopsy_env, policy_kwargs = policy_kwargs, tensorboard_log = LOG_DIR, device = torch.device('cpu'))
        
        # Obtain IL model weights from SL training 
        model = agent.policy 
        model.load_state_dict(torch.load(MODEL_PATH, map_location = torch.device('cuda'))) #, map_location = torch.device('cpu')))
        
        # Load as agent with loaded weights!!! 
        agent.policy = model 
    
    else:
        # load agent directly 
        agent = PPO.load(MODEL_PATH, Biopsy_env, device = torch.device('cuda'))
    
    # TODO : Change this to number of lesions in test dataset : Number of dataset is the number of lesions you have, not just number of patients 
    # Alternatively, just use number of patients NUM_DATASET = 100 
    # if MODE == 'train':
    #     NUM_DATASET = 966 
    # elif MODE == 'val':
    #     NUM_DATASET = 141 
    # else:
    #     NUM_DATASET = 275 
        
    # print(f"Num dataset : {NUM_DATASET}")
    
    # TO FIND NUMBER OF LESIONS IN YOUR DATASET ,YOU CAN UNCOMMENT THIS CODE: 
    # print number of lesions * each dataset
    # nuimber of lesiosn in test set : 972
    # total_num_lesions = 0
    # total_num_lesions_v2 = 0 
    # for idx, (mri_vol, prostate_mask, lesion_mask, sitk_img_path , num_lesions, patient_name) in enumerate(PS_dataset):
    #     lesion_labeller = LabelLesions()
    #     tumour_centroids, num_lesions_v2, tumour_statistics, multiple_label_img = lesion_labeller([sitk_img_path,'a'])
    #     print(f"num lesions {num_lesions} sitk : {num_lesions_v2}")
    #     total_num_lesions += num_lesions
    #     total_num_lesions_v2 += num_lesions_v2
    #     print(f"Total number lesions: {total_num_lesions}")
    

    #NUM_DATASET = 100
    print(f"Use RL {USE_RL} file name : {args.file_name}")
    
    NUM_DATASET = 100
    average_val_reward, std_episode_reward, avg_episode_len, avg_hit_rate, avg_ccl_corr, avg_efficiency, avg_hit_threshold, all_ccl, all_sizes = evaluate_agent(agent, NUM_DATASET, Biopsy_env, evaluate_mode = evaluate_mode, file_name = args.file_name, use_rl = USE_RL, log_dir = RESULTS_DIR)
    ccl_all = np.concatenate(all_ccl)
    sizes_all = np.concatenate(all_sizes)
    
    # FOR VISUALISING CCL COEFFICIENT
    # plt.figure()
    # plt.scatter(sizes_all, ccl_all)
    # plt.xlabel("Lesion size (number of pixels)")
    # plt.ylabel("CCL")
    # plt.title("CCL vs lesion size")    
    
    