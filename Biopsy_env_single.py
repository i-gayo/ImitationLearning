import gym
from gym import spaces
import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.twodim_base import mask_indices 
from scipy.spatial.distance import cdist as cdist  
from scipy.interpolate import interpn
import copy
import os 
from stable_baselines3 import PPO
from supersuit import frame_stack_v1
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper

from utils_data import *
#Import all dataloader functions 
from Prostate_dataloader import * 
from stable_baselines3.ppo.policies import CnnPolicy#, MultiInputPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
import torch 
import time 

from PIL import Image, ImageDraw, ImageFont
from environment_utils import * 

import matplotlib
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

# class NewFeatureExtractor(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#         Correspods to the number of unit for the last layer.
#     """

#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, multiple_frames = False, num_multiple_frames = 3):
        
#         super(NewFeatureExtractor, self).__init__(observation_space, features_dim)
#         # Assumes CxHxW images (channels first)
#         # Re-ordering will be done by pre-preprocessing or wrapper

#         #num_input_channels = observation_space.shape[-1] #rows x cols x channels 
#         #num_multiple_frames = 3
#         #num_multiple_frames = observation_space.shape[-1]
#         #self.num_multiple_frames = num_multiple_frames

#         num_channels = 5 
#         self.cnn_layers = nn.Sequential(

#             # First layer like resnet, stride = 2
#             nn.Conv3d(num_channels, 32, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(32),
#             nn.ReLU(),

#             # Apply pooling layer in between 
#             nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1),

#             nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm3d(64),
#             nn.ReLU(),

#             # Apply pooling layer in between 
#             nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1),

#             nn.Conv3d(64, 128, kernel_size=3, stride=1),
#             nn.BatchNorm3d(128),
#             nn.ReLU()
#         )

#         #Flatten layers 
#         self.flatten = nn.Flatten()

#         # Compute shape by doing one forward pass
#         with torch.no_grad():
#             all_layers = nn.Sequential(self.cnn_layers, self.flatten)
            
#             #observation_space_shuffled = np.transpose(observation_space.sample(), [2, 1, 0])
#             #n_flatten = all_layers(torch.as_tensor(observation_space_shuffled[None]).float()).shape[1]
#             #processed_obs_space = self._pre_process_image(torch.zeros))).float()
#             processed_obs_space = torch.zeros([1, 5, 100, 100, 24])
#             n_flatten = all_layers(processed_obs_space).shape[1]  

#         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
#         #observations = self._pre_process_image(observations)
#         observations = observations.float() 
#         output = self.cnn_layers(observations)
#         output = self.flatten(output)
        
#         return self.linear(output)

#     def _pre_process_image(self, images):
#         """ 
#         A function that switches the dimension of image from row x col x channel -> channel x row x colmn 
#         and addeds a dimension along 0th axis to fit network 
#         """ 
#         #print(f'Image size {images.size()}')
#         image = images.clone().detach().to(torch.uint8)#.squeeze()
#         if len(np.shape(images)) == 5:
#             image = image.squeeze()
#         split_channel_image = torch.cat([torch.cat([image[j,:,:,i*25:(i*25)+25].unsqueeze(0) for i in range(3)]).unsqueeze(0) for j in range(image.size()[0])])#.clone().detach().to(torch.uint8)
#         #split_channel_image = torch.cat([torch.cat(torch.tensor_split(image[i,:,:,:].unsqueeze(0), self.num_multiple_frames, dim=3)).unsqueeze(0) for i in range(image.size()[0])])
#         #processed_image = image.permute(0, 3,2,1)
#         #processed_image = torch.unsqueeze(processed_image, dim= 0)
        
#         # Turn image from channel x row x column -> channel x row x column x depth for pre-processing with 3D layers 

#         return split_channel_image

class SaveOnBestTrainingReward_single(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingReward_single, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.best_mean_reward_std = np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          df_training = load_results(self.log_dir)
          x, y = ts2xy(df_training, 'timesteps')
        
          efficiency = np.nanmean(df_training.efficiency.values)
          ccl_corr = np.nanmean(df_training.ccl_corr_online.values)
          hit_rate = np.nanmean(df_training.hit_rate.values)
          num_needles = np.nanmean(df_training.num_needles.values)
          num_needles_hit = np.nanmean(df_training.num_needles_hit.values)
          #ccl_plots = df_training.ccl_plots.values
          lesion_sizes = df_training.all_lesion_size.values
          ccl_vals = df_training.all_ccl.values

          #Convert lesion size and ccl vals to plot 
          lesion_list = np.concatenate([ast.literal_eval(lesion) for lesion in lesion_sizes])
          ccl_list = np.concatenate([ast.literal_eval(ccl) for ccl in ccl_vals])
          #figure_plot = plt.figure()
          #plt.scatter(lesion_list , ccl_list)
          #plt.xlabel("Lesion sizes (number of voxels)")
          #plt.ylabel("CCL (mm)")
          #ccl_fig = plt.gcf()
          
          self.logger.record('metrics/ccl_coef', ccl_corr)
          self.logger.record('metrics/hit_rate', hit_rate)
          self.logger.record('metrics/efficiency' , efficiency)
          self.logger.record('needles/num_needles' , num_needles)
          self.logger.record('needles/num_needles_hit' , num_needles_hit)
          # Plot last data (most updaetd CCL batch size)
          #self.logger.record("metrics/ccl_plots", Figure(ccl_fig, close=True), exclude=("stdout", "log", "json", "csv"))
          plt.close()

          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.nanmean(y[-self.check_freq:])
              std_reward = np.nanstd(y[-self.check_freq:])

              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} +/- {self.best_mean_reward_std:.2f} \
                - Last mea4n reward per episode: {mean_reward:.2f} +/- {std_reward:.2f}")

              # Save the model if the mean reward is better than the previously saved mean reward 
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  self.best_mean_reward_std = std_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True

def compute_needle_efficiency(num_needles_hit, num_needles_fired):
  """
  A function which computes the needle efficiency ie 
  ratio between : how many needles actually hit the lesions / 
                  how many needles were fired in total 
  """
  pass 

def round_to_05(val):
    """
    A function that rounds to the nearest 5mm
    """
    #rounded_05 = round(val * 2) / 2
    if torch.is_tensor(val):
      rounded_05 = 5 * torch.round(val / 5)
    else:
      rounded_05 = 5 * round(val / 5)
    return rounded_05

def online_compute_coef(x_n1, y_n1, xbar = 0, ybar = 0, Nn = 0, Dn=0, En=0, n=0):
    
    """
    A function which computes the online CCL coeff given a new value
    """
    
    xbar_n1 = xbar + ((x_n1 - xbar)/(n+1))
    ybar_n1 = ybar + ((y_n1 - ybar)/(n+1))

    N_n1 = Nn + (x_n1 - xbar)*(y_n1 - ybar_n1)
    D_n1 = Dn + (x_n1 - xbar)*(x_n1 - xbar_n1)
    E_n1 = En + (y_n1 - ybar)*(y_n1 - ybar_n1)

    r = N_n1 / (np.sqrt(D_n1) * np.sqrt(E_n1))

    return r, xbar_n1, ybar_n1, N_n1, D_n1, E_n1

def add_num_needles_left(grid_array, num_needles_left):
    
  """
  A function that adds num needles left to the grid array as text

  Note:
  --------
  First converts the array to an image, then adds text to the image which number of needles left 

  """

  #only copy previous points not the text
  #grid_only = np.zeros_like(grid_array)
  #grid_only[0:90,0:90] = grid_array[0:90,0:90]
  grid_img = Image.fromarray(np.uint8(grid_array*255))

  #Add text to image with num needles left
  text_str = "Needles left:" + str(num_needles_left)
  draw_grid_img = ImageDraw.Draw(grid_img)
  draw_grid_img.text((3, 90), text_str, fill = (200))

  #Convert image back to array and normalise between 0 to 1
  grid_img_array = np.array(grid_img)
  normalised_img = (grid_img_array - np.min(grid_img_array)) / (np.max(grid_img_array) - np.min(grid_img_array))

  return normalised_img 


class TemplateGuidedBiopsy_single(gym.Env):
    """Biopsy environment for multiple patients, observing only single lesions at a time """

    metadata = {'render.modes': ['human']}

    def __init__(self, DataSampler, obs_space = 'images', results_dir = 'test', env_num = '1', reward_fn = 'ccl', \
    miss_penalty = 2, terminating_condition = 'max_num_steps', train_mode = 'train', device = 'cpu', max_num_steps = 100, \
      penalty = 5, deform = False, deform_scale = 0.1, deform_rate = 0.25, start_centre = False):

        """
        Actions : delta_x, delta_y, z (fire or no fire or variable depth)
        """
        super(TemplateGuidedBiopsy_single, self).__init__()

        self.obs_space = obs_space

        ## Defining action and observation spaces
        self.action_space = spaces.Box(low = -1, high= 1, shape = (3,), dtype = np.float32)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5, 100, 100, 24), dtype=np.float64)

        # Load dataset sampler 
        self.DataSampler = DataSampler
        
        # starting condition on grid (within centre of template grid)
        img_data = self.sample_new_data()

        # Defining variables 
        self.done = False 
        self.reward_fn = reward_fn
        self.num_needles = 0 
        self.max_num_needles = 4 #* self.num_lesions
        self.num_needles_per_lesion = np.zeros(self.num_lesions)
        self.all_ccl = [] 
        self.all_sizes = []
        self.max_num_steps_terminal = max_num_steps
        print(f"max num steps terminal : {self.max_num_steps_terminal}")
        self.step_count = 0 
        self.device = device
        self.previous_ccl_corr = 0
        self.hit_rate_threshold = 0.6
        self.previous_ccl_online = 0 
        self.penalty_reward = penalty 
        self.terminating_condition = terminating_condition
        self.needle_penalty = miss_penalty
        self.start_centre = start_centre
        self.deform = deform 
        self.deform_rate = deform_rate
        self.deform_scale = deform_scale 
        self.all_norm_ccl = [] 
        self.num_needles_hit = 0 
        
        # Defining deformation transformer to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")
        self.random_transform = GridTransform(grid_size=[8,8,4], interp_type='linear', volsize=[100,100,24], batch_size=1, device=device)

        # Initialise state
        initial_obs, starting_pos, current_pos = self.initialise_state(img_data, start_centre)
        self.initial_obs = initial_obs 
        self.current_obs = initial_obs 

        # Save starting pos for next action movement 
        self.current_needle_pos = starting_pos # x and y only; 
        self.current_pos = current_pos # x,y,z depth current poss 
        

        # Correlation statistics
        self.r = 0
        self.xbar = 0
        self.ybar = 0
        self.N = 0
        self.D = 0
        self.E = 0
        #self.timer_online = 0 

        # Statistics about data 
        if train_mode == 'train':
            self.num_data = 402 # 105
        elif train_mode == 'test':
            self.num_data = 115 # 30 
        else:
            self.num_data = 58 # 15

        # Add a patient counter 
        self.patient_counter = 0 

        # Visualisation files
        self.file_output_actions = os.path.join(results_dir, \
          ('_output_actions_' + train_mode + '_' + env_num + '.csv'))

        self.file_patient_names = os.path.join(results_dir, \
          ('_patient_names' + train_mode + '_' + env_num + '.csv'))

        with open(self.file_output_actions, 'w') as fp: 
          fp.write('''\
          x_grid, y_grid, depth, reward
          ''')

        with open(self.file_patient_names, 'w') as fp: 
          fp.write('''\
          Patient_name
          ''')
          fp.write('\n')

        with open(self.file_patient_names, 'a') as fp: 
          fp.write(str(self.patient_name[0]))
          fp.write('\n')

    def step(self, action):
        
        """
        Determines how actions affect environment
        """

        self.step_count += 1

        #if self.patient_name[0] == 'Patient479592532_study_1.nii.gz':
        #  print(f"Step count : {self.step_count}")

        ### 1. Un-normalise actions : normalised between (-1,1)
        z_unnorm = action[2] + 1 #un-normalise from (-1,1) -> 0,2 where 0 is non-fired, 1 is apex, 2 is base 
        if z_unnorm <= -0.33: 
          needle_fired = False
          z_depth = 0 
        elif ((z_unnorm > -0.33) and (z_unnorm <= 0.33)): # apex
          needle_fired = True 
          z_depth = 1
        else: # base 
          needle_fired = True 
          z_depth = 2 
    
        ### 2. Move current template pos according to action_x, action_y -> DOUBLE CHECK THIS 
        grid_pos, same_position, moves_off_grid = self.find_new_needle_pos(action[0], action[1])
        #print(f'Grid pos : {grid_pos}')
        self.current_needle_pos = grid_pos 
        
        # Append z depth to current pos
        self.current_pos = np.append(copy.deepcopy(grid_pos), z_depth)

        ### 3. Update state, concatenate grid_pos to image volumes 
        new_obs = self.obtain_new_obs(self.current_pos)
        self.current_obs = new_obs 

        #new_grid_array = self.create_grid_array(grid_pos[0], grid_pos[1], needle_fired, self.grid_array, display_current_pos = True)

        ### TODO: CHAGNE REWARD , CCL AND HR 

        ### 4. Compute reward and CCL if needle fired and append to list of CCL_coeff
        needle_hit = False 

        if needle_fired:

            self.num_needles += 1

            # Check if previously fired here or not 
            fired_same_position = (self.firing_grid[grid_pos[1]  + 50, grid_pos[0] + 50] == 1)

            ## For debugging purpose only -> check which points in the grid are hit by needles
            # Save firing grid position as 1  
            y_grid_pos = grid_pos[1] + 50
            x_grid_pos = grid_pos[0] + 50
            self.firing_grid[y_grid_pos:y_grid_pos+ 2, x_grid_pos] = 1
            self.firing_grid[y_grid_pos - 1 :y_grid_pos, x_grid_pos] = 1
            self.firing_grid[y_grid_pos, x_grid_pos:x_grid_pos + 2 ] = 1
            self.firing_grid[y_grid_pos, x_grid_pos - 1 : x_grid_pos] = 1

            # 4. Obtain needle sample trajectory, compute CCL using ground truth masks
            needle_traj, intersect_vol, ground_truth_z = self.compute_needle_traj(grid_pos[0], grid_pos[1], z_depth) #note grid_pos[0] and grid_pos[1] need to correspond to image coords
            ccl, ccl_approx, max_ccl = self.compute_ccl(intersect_vol)
            norm_ccl = ccl / max_ccl #normalised ccl

            # only add ccl to fired needles; don't consider non-fired needles 
            self.all_ccl.append(ccl)
            self.all_norm_ccl.append(norm_ccl)
            
            ## Only considering single lesions, so no need to do single 
            # # If two lesions were hit by the same needle, append each ccl and size separately
            # two_lesions_hit = (type(ccl) == list)

            # if two_lesions_hit:
              
            #   for i in range(len(ccl)):
            #     self.all_ccl.append(ccl[i])
            #     self.all_sizes.append(self.tumour_statistics['lesion_size'][lesion_idx[i]])
            #     self.num_needles_per_lesion[lesion_idx[i]] += 1
              
            #   #Increase successful needle count
            #   self.num_needles_hit +=1 
            #   needle_hit = True 

            # No lesion hit therefore no lesion size 
            if ccl == 0:
                self.all_sizes.append(0)
                needle_hit = False 

            else:
                needle_hit = True 
                self.all_sizes.append(self.tumour_statistics['lesion_size'][self.lesion_idx])
                self.num_needles_per_lesion[self.lesion_idx] += 1
              
                #Increase successful needle count
                self.num_needles_hit +=1 
                needle_hit = True
          
            # Compute CCL coefficient online 
            n_val = len(self.all_ccl)
            ccl_corr_online, self.xbar, self.ybar, self.N, self.D, self.E = online_compute_coef(self.all_ccl[-1], self.all_sizes[-1], self.xbar, self.ybar, self.N, self.D, self.E, n = n_val)
            
            # from nan to 0 ccl corr 
            if np.isnan(ccl_corr_online):
              ccl_corr_online = 0 
            self.previous_ccl_online = ccl_corr_online
            
            # Check if needle hits the prostate 
            needle_hits_outside_prostate = self.check_needle_hits_outside_prostate(needle_traj)

        else:
            # No needle fired, so no ccl obtained
            ccl = 0  
            needle_hit = False 
            norm_ccl = 0

            #Use previous ccl correlation as no update to ccl values 
            ccl_corr_online = self.previous_ccl_online   
            needle_hits_outside_prostate = False 
            needle_traj, intersect_vol, ground_truth_z = self.compute_needle_traj(grid_pos[0], grid_pos[1], z_depth)
            needle_traj = np.zeros_like(self.img_data['mri_vol'])
        
        # Add number of needles left as additional info 

        needles_left = self.max_num_needles - self.num_needles
        #new_grid_array = add_num_needles_left(new_grid_array, needles_left)
          
        #new_obs = self.obtain_obs(new_grid_array)
        #new_obs = self.obtain_obs_wneedle(new_grid_array, needle_traj)

        ### 5. Check if episode terminates episode if hit rate threshold is reached or max_num_steps is reached 
        
        # Commpute statistics 
        #all_lesions_hit = np.all(self.num_needles_per_lesion >= 1)
        if self.num_needles == 0:
          hr = 0 
        else: 
          hr = self.num_needles_hit / self.num_needles # how many needles fired hit lesion 

        hit_threshold_reached = (self.num_needles_hit >= self.max_num_needles) # when num needles hit lesion > 4 
        max_num_needles_fired = (self.num_needles >= self.max_num_needles)  # when num needles fired > 4
        max_num_steps_reached = (self.step_count >= self.max_num_steps_terminal) 
        more_than_5 = self.num_needles_per_lesion[self.lesion_idx] >= 5 # if more htan 5 needles hit, terminate episode 

        #agent_hit_rate = np.mean((self.num_needles_per_lesion >= 2)) # how many lesions are hit at least twice 
        #hit_threshold_reached = agent_hit_rate >= self.hit_rate_threshold # if hit rate threshold is reached 
        #max_num_needles_fired = (self.num_needles >= self.max_num_needles) 
        #max_num_steps_reached = (self.step_count >= self.max_num_steps_terminal) 
        #max_num_steps_reached = (self.step_count >= (self.max_num_needles + 10)) 

        # Compute efficiency 
        if self.num_needles == 0:
          efficiency = 0 
          hr = 0 
        else:
          efficiency = self.num_needles_hit / self.num_needles
          hr = self.num_needles_hit / self.num_needles
          
        # Terminate depending on whether max num steps are reached OR if hit threshold is reached 
        if self.terminating_condition == 'max_num_steps':
          terminate = max_num_steps_reached 
        elif self.terminating_condition == 'hit_threshold':
          terminate = max_num_steps_reached or hit_threshold_reached 
        elif self.terminating_condition == 'max_num_needles_fired':
          terminate = max_num_steps_reached or max_num_needles_fired
        else: 
          moves_off_same_pos = (moves_off_grid and same_position)
          if moves_off_same_pos:
            print(f"Tries to move off grid again, so terminate early.")
          terminate = max_num_steps_reached or more_than_5 or moves_off_same_pos 


        if terminate: #or hit_threshold_reached:
            done_new_patient = True 
            self.patient_counter += 1
        else:
            done_new_patient = False 
            self.current_ccl_plot = None

        if done_new_patient:
          figure_plot = plt.figure()
          plt.scatter(self.all_sizes , self.all_ccl)
          plt.xlabel("Lesion sizes (number of voxels)")
          plt.ylabel("CCL (mm)")
          self.current_ccl_plot = plt.gcf()
          plt.close()

        # 6. Compute reward function 

        #reward = self.compute_reward_simple_reward(done_new_patient, ccl_corr_online, agent_hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid)
        if self.reward_fn == 'classify_reward':
          reward = self.compute_reward_classify(needle_fired, needle_hit, needle_hits_outside_prostate, ground_truth_z, z_depth)
        elif self.reward_fn == 'penalty':
          reward = self.compute_reward_penalty(needle_fired, needle_hit, needle_hits_outside_prostate, ground_truth_z, z_depth, terminate)
        elif self.reward_fn == 'reward':
          reward = self.compute_reward(needle_fired, needle_hit, needle_hits_outside_prostate)
        else:
          # compute based on metrics 
          reward = self.compute_reward_metrics(hr, needle_fired, needle_hit, needle_hits_outside_prostate, terminate, self.reward_fn)

        # Add penalty for moves off grid
        if moves_off_grid: 
          reward -= 10 # penalise for moving off grid continuously 
        
        if moves_off_same_pos:
          reward -= 100 # penalise for moving off and repeating same action. Terminal reward is 100. 
          
        scaled_lesion_centre = (self.tumour_centroids[self.lesion_idx] + self.prostate_centroid) * np.array([0.5, 0.5, 0.25]) - np.array([50, 50, 0])
        print(f"Needle hit : {needle_hit} Reward : {reward} \n \
              Lesion centre : {scaled_lesion_centre} POSITION : {self.current_pos} Action: {action}")
        
        # Save ccl_corr as new previous_ccl_corr
        self.previous_ccl_corr = ccl_corr_online

        # 6. Compute saving statistics and actions print(f"Reward : {reward}")
        saved_actions = np.array([grid_pos[0], grid_pos[1], self.max_needle_depth*int(needle_fired), reward])

        with open(self.file_output_actions, 'a') as fp:
          np.savetxt(fp, np.reshape(saved_actions, [1,-1]), '%s', ',')

        info = {'num_needles_per_lesion' : self.num_needles_per_lesion, 'all_ccl' : self.all_ccl,\
             'all_lesion_size' : self.all_sizes, 
             'ccl_corr' : ccl_corr_online, 'hit_rate' : hr, \
               'new_patient' : done_new_patient, 'ccl_corr_online' : ccl_corr_online, \
                'efficiency' : efficiency,'num_needles' : self.num_needles, \
                  'max_num_needles' : self.max_num_needles, 'num_needles_hit' : self.num_needles_hit, \
                    'firing_grid' : self.firing_grid, 'hit_threshold_reached' : hit_threshold_reached, \
                      'lesion_mask' : self.img_data['tumour_mask'], 'all_norm_ccl' : self.all_norm_ccl, \
                        'norm_ccl' : norm_ccl, 'current_pos' : self.current_pos, 'needle_hit' : needle_hit} 

        # Reset ccl statistics after going through entire dataset ie zero ccl and all_lesion_sizes
        if self.patient_counter  >= self.num_data:
            self.reset_ccl_statistics()
            self.all_ccl = [] 
            self.all_sizes = [] 
            self.patient_counter = 0 
            #bonus_reward = 100 * ccl_corr_online
            #reward += bonus_reward 

        return new_obs, reward, done_new_patient, info

    def reset_ccl_statistics(self):
        
        img_data = self.sample_new_data()
        initial_obs, starting_pos, current_pos = self.initialise_state(img_data, self.start_centre)
        self.current_obs = initial_obs
        #print(f"Starting pos : {starting_pos}")

        # Save starting pos for next action movement 
        self.current_needle_pos = starting_pos 
        self.current_pos = current_pos

        # Defining variables 
        self.done = False 
        self.num_needles = 0 
        self.num_needles_hit= 0 
        self.max_num_steps = 4 * self.num_lesions
        self.num_needles_per_lesion = np.zeros(self.num_lesions)
        self.all_ccl = [] 
        self.all_norm_ccl = [] 
        self.all_sizes = [] 
        self.step_count = 0 
        self.patient_counter = 0 

        # Correlation statistics
        self.r = 0
        self.xbar = 0
        self.ybar = 0
        self.N = 0
        self.D = 0
        self.E = 0
        #self.timer_online = 0 

        #Add line to actions file to indicate a new environment has been started
        with open(self.file_output_actions, 'a') as fp:
          fp.write('\n')

        with open(self.file_patient_names, 'a') as fp: 
          fp.write(str(self.patient_name[0]))
          fp.write('\n')

        #if self.obs_space != 'images':
        #  initial_obs = {'img_volume': initial_obs, 'num_needles_left' : self.max_num_needles - self.num_needles}

        return initial_obs  # reward, done, info can't be included

    def reset(self):
        
        img_data = self.sample_new_data()
        initial_obs, starting_pos, current_pos = self.initialise_state(img_data, self.start_centre)
        self.current_obs = initial_obs
        
        #if self.obs_space != 'images':
        #  initial_obs = {'img_volume': initial_obs, 'num_needles_left' : self.max_num_needles - self.num_needles}
        #print(f"Starting pos : {starting_pos}")

        # Save starting pos for next action movement 
        self.current_needle_pos = starting_pos 
        self.current_pos = current_pos

        # Defining variables 
        self.done = False 
        self.num_needles = 0 
        self.num_needles_hit = 0 
        self.max_num_steps = 4 * self.num_lesions
        self.max_num_needles = 4 #* self.num_lesions
        self.num_needles_per_lesion = np.zeros(self.num_lesions)
        #self.all_ccl = [] 
        #self.all_sizes = [] 
        self.step_count = 0 
        self.all_norm_ccl = [] 
        #self.timer_online = 0 

        #Add line to actions file to indicate a new environment has been started
        with open(self.file_output_actions, 'a') as fp:
          fp.write('\n')

        with open(self.file_patient_names, 'a') as fp: 
          fp.write(str(self.patient_name[0]))
          fp.write('\n')

        return initial_obs  # reward, done, info can't be included

    def render(self, mode='human'):
        pass 
    
    def close (self):
        pass 

    """ Helper functions """ 

    def create_needle_vol(self, current_pos):

      """
      A function that creates needle volume 100 x 100 x 24 

      Parameters:
      -----------
      current_pos : current position on the grid. 1 x 3 array delta_x, delta_y, delta_z ; assumes in the range of (-30,30) ie actions are multipled by 5 already 

      Returns:
      -----------
      neeedle_vol : 100 x 100 x 24 needle vol for needle trajectory 

      """
      needle_vol = np.zeros([100,100,24])

      x_idx = current_pos[0]#*5
      y_idx = current_pos[1]#*5

      #Converts range from (-30,30) to image grid array
      x_idx = (x_idx) + 50
      y_idx = (y_idx) + 50

      x_grid_pos = int(x_idx)
      y_grid_pos = int(y_idx)

      depth_map = {0 : 1, 1 : int(0.5*self.max_depth), 2 : self.max_depth}
      depth = depth_map[int(current_pos[2])]

      needle_vol[y_grid_pos-1:y_grid_pos+ 2, x_grid_pos-1:x_grid_pos+2, 0:depth ] = 1

      return needle_vol 

    def obtain_new_obs(self, current_pos):
      """
      Obtains observations, given the new grid position 

      Parameters:
      ----------
      current_pos : 3 x 1 array (x,y,z) where z is (0,1,2) where 0 i snon-fired, 1 is apex and 2 is base 
      
      Returns:
      ----------
      obs : 5 x 100 x 100 x 25 observations 
      
      """

      needle_vol = self.create_needle_vol(current_pos)
      
      # Obtain old needle stack from T-2 : T
      old_needle = self.current_obs[-3:,:,:,:]

      # Replace needle stack. Move T-1 -> T-2, T -> T-1 and new needle as T
      new_needle = torch.zeros_like(old_needle)
      new_needle[0:2, :, :, :] = old_needle[1:, :,:, :]
      new_needle[-1,:,:,:] = torch.tensor(needle_vol)

      # Obtain new obs : replace needle stack with needle stack 
      new_obs = copy.deepcopy(self.current_obs)
      new_obs[-3:, :, :] = new_needle 

      return new_obs
      
    def obtain_obs(self, template_grid):
        """
        Obtains observations from current template grid array and stacks them 

        Notes:
        ----------
        Down-samples and only obtains every 2 pixels for CNN efficiency 

        """

        prostate_vol = self.noisy_prostate_vol[:, :, :] #prostate = 1
        tumour_vol = self.noisy_tumour_vol[:, :, :] * 2 #tumour = 2
        combined_tumour_prostate = prostate_vol + tumour_vol

        #Convert intersection to just be lesion (avoid overlap)
        combined_tumour_prostate[combined_tumour_prostate >= 2] = 2

        new_obs = np.concatenate([np.expand_dims(template_grid, axis = 2), combined_tumour_prostate], axis = 2)
        new_obs = new_obs * 0.5

        return new_obs
    
    def obtain_obs_wneedle(self, template_grid, needle_mask):

        def add_grid(grid, vol):
          """
          A function that adds the grid to the images 
          """
          grid = np.expand_dims(grid, axis = 2)
          combine_grid = np.concatenate((grid, vol), axis = 2)

          return np.expand_dims(combine_grid, axis = 3)
        
        prostate_vol = add_grid(template_grid, self.noisy_prostate_vol[:, :, :])
        tumour_vol = add_grid(template_grid,self.noisy_tumour_vol[:, :, :])
        needle_mask_vol = needle_mask[0::2,0::2, 0::4]
        needle_vol = add_grid(template_grid, needle_mask_vol)

        combined_vol = np.concatenate((prostate_vol, tumour_vol, needle_vol), axis = -1) # Stack volumes on top of each other
        combined_vol = np.transpose(combined_vol, [3, 0, 1, 2])
        
        return combined_vol 

    def sample_new_data(self):
        """
        Obtains new patient data once an episode terminates 
        
        """

        
        self.firing_grid = np.zeros([100,100])

        (mri_vol, prostate_mask, tumour_mask, tumour_mask_sitk, rectum_pos, self.patient_name) = self.DataSampler.sample_data()
        #print(f"Patient name: {self.patient_name}")
        #Turn from tensor to numpy array for working with environment
        mri_vol = np.squeeze(mri_vol.numpy())
        tumour_mask = np.squeeze(tumour_mask.numpy())
        prostate_mask = np.squeeze(prostate_mask.numpy())
        rectum_pos = np.squeeze([rectum_p.numpy() for rectum_p in rectum_pos])

        #Initialising variables to save into model 
        self.img_data = {'mri_vol' : mri_vol, 'tumour_mask' : tumour_mask, 'prostate_mask': prostate_mask}
        self.volume_size = np.shape(mri_vol)
        self.rectum_position = rectum_pos #x,y,z 

        #Initialising needle sample length : 10mm 
        self.L = 15.0 #np.random.uniform(low = 5.0, high = 15.0)

        #Obtain bounding box of prostate, tumour masks (for tumour this is a bounding sphere) 
        self.bb_prostate_mask, self.prostate_centroid = self._extract_volume_params(prostate_mask, which_case= 'Prostate')
        self.max_needle_depth = np.max(np.where(self.img_data['prostate_mask'] == 1)[-1]) #max z depth with prostate present using whole volume 
        self.max_depth = int(self.max_needle_depth /4) # downsamples image volume so / 4 
        #self.max_depth = np.max(np.where(self.img_data['prostate_mask'][::2,::2,::4] == 1)[-1]) #max z depth with prostate present

        #Obtain image coordinates centred at the rectum 
        self.img_coords = self._obtain_vol_coords()

        lesion_labeller = LabelLesions()
        self.tumour_centroids, self.num_lesions, self.tumour_statistics, self.multiple_label_img = lesion_labeller(tumour_mask_sitk)
        self.tumour_centroids -= self.prostate_centroid # Centre coordinates at prostate centroid 
        #print(f"Tumour centroids {self.tumour_centroids}")
        self.tumour_projection = np.max(self.multiple_label_img, axis = 2)
        return self.img_data 

    def initialise_state(self, img_vol, start_centre = False):

        """
        A function that initialises the state of the environment 
        
        Returns:
        ----------
        :state: 5 x 100 x 100 x 24 stack of observations 

        Notes:
        ---------
        - prostate mask 100 x 100 x 24
        - lesion mask 100 x 100 x 24 
        - needle masks 3 x 100 x 100 x 24 
        """

        
        # 3. Obtain noisy prostate and tumour masks (add reg noise )
        self.noisy_prostate_vol, tre_prostate = self.add_reg_noise(img_vol['prostate_mask'], tre_ = 3)
        
        # sample single lesion mask 
        #lesion_idx = np.random.choice(np.arange(1, self.num_lesions+1))
        #self.lesion_idx = (lesion_idx - 1) # 0 index, whilst masks are 1-indexed 
        
        # Sample only lesion 1 
        lesion_idx = 1
        self.lesion_idx = lesion_idx - 1 
        lesion_mask = self.separate_masks(self.multiple_label_img, lesion_idx)

        print(f"Lesion centre : {self.tumour_centroids[self.lesion_idx] + self.prostate_centroid}")
        # Note : observation should now be just single lesion mask, instead of all of them!!! corrected bug 
        # previous : self.add_reg_noise(img_data['tumour_mask]) which has all of hte masks, instead of single 

        self.true_tumour_vol = lesion_mask 
        self.noisy_tumour_vol, tre_tumour = self.add_reg_noise(lesion_mask, tre_ = 4) # more tre for tumour
        
        # 1. Initialise starting point on template grid : within centre box of grid 
        all_possible_points  = np.arange(-15, 20, 5)

        if start_centre: 
          starting_x = 0
          starting_y = 0 
        else:
          # start from centre of lesion
          starting_x, starting_y = self.start_lesion_bb()
          #print(f"starting_x,y : [{starting_x}, {starting_y}]")

          #starting_x, starting_y = np.random.choice(all_possible_points, 2)

        # 2. Obtain grid of initial needle starting position 
        grid_array = self.create_grid_array(starting_x, starting_y, needle_fired = False, display_current_pos = True)
        self.grid_array = copy.deepcopy(grid_array)

        # Include number of needles left as additional info on image 
        grid_array = add_num_needles_left(grid_array, self.max_num_needles)
        starting_points = np.array([starting_x, starting_y]) 

        # Obtain needle obs 
        starting_z = 0
        current_pos = np.array([starting_x, starting_y, starting_z])
        needle_obs = self.create_needle_vol(current_pos)
        needle_stack = torch.tensor(np.array([np.zeros([100,100,24]), np.zeros([100,100,24]), needle_obs])) # t = -2 : t = 0 

        # deform 
        obs_volume = torch.stack((torch.tensor(self.noisy_tumour_vol), torch.tensor(self.noisy_prostate_vol))).unsqueeze(0)
        
        if self.deform: 
          obs_volume = self.apply_deforms(obs_volume.to(torch.device("cuda")), self.deform_rate, self.deform_scale)
          obs_volume.to(torch.device("cpu"))
        
        #obs = self.obtain_obs_wneedle(grid_array, np.zeros_like(self.img_data['mri_vol']))

        obs = torch.cat((torch.tensor(self.noisy_tumour_vol).unsqueeze(0), torch.tensor(self.noisy_prostate_vol).unsqueeze(0), needle_stack), axis = 0)

        return obs, starting_points, current_pos

    def start_lesion_bb(self):
      """
      
      # Initialises the needle starting point within bounding box of lesions


      """
      #TODO : find grid points within bb of tumour 
      lesion_coords = np.where(self.true_tumour_vol == 1)
      min_coords = np.min(lesion_coords, axis = 1)
      max_coords = np.max(lesion_coords, axis = 1)
      
      # randomly sample from bb of lesion 
      x_pos = int(np.random.uniform(min_coords[1], max_coords[1]))
      y_pos = int(np.random.uniform(min_coords[0], max_coords[0]))
      
      # find closest lesion point for grid map
      x_grid = (torch.arange(-30,35,5))*2 + self.prostate_centroid[0]
      y_grid = (torch.arange(-30,35,5))*2 + self.prostate_centroid[1]
      closest_x = x_grid[torch.argmin(torch.abs(x_grid - x_pos))]
      closest_y = y_grid[torch.argmin(torch.abs(y_grid - y_pos))]
      
      #TODO : randomly sample grid points within tumoru 
      x_grid_pos = int((closest_x - self.prostate_centroid[0])/2)
      y_grid_pos = int((closest_y - self.prostate_centroid[1])/2)
      
      print(f"starting needle img coords : {closest_x}, {closest_y}")
      
      return torch.tensor([x_grid_pos, y_grid_pos])
    
    def get_start_grid_pos(self):
      """
      
      # Initialises the needle starting point within bounding box of lesions


      """
      #TODO : find grid points within bb of tumour 
      lesion_coords = np.where(self.true_tumour_vol == 1)
      min_coords = np.min(lesion_coords, axis = 1)
      max_coords = np.max(lesion_coords, axis = 1)
      
      # randomly sample from bb of lesion 
      x_pos = int(np.random.uniform(min_coords[1], max_coords[1]))
      y_pos = int(np.random.uniform(min_coords[0], max_coords[0]))
      
      # find closest lesion point for grid map
      x_grid = (torch.arange(-30,35,5))*2 + self.prostate_centroid[0]
      y_grid = (torch.arange(-30,35,5))*2 + self.prostate_centroid[1]
      closest_x = x_grid[torch.argmin(torch.abs(x_grid - x_pos))]
      closest_y = y_grid[torch.argmin(torch.abs(y_grid - y_pos))]
      
      #TODO : randomly sample grid points within tumoru 
      x_grid_pos = int((closest_x - self.prostate_centroid[0])/2)
      y_grid_pos = int((closest_y - self.prostate_centroid[1])/2)
      
      print(f"starting needle img coords : {closest_x}, {closest_y}")
      
      return torch.tensor([x_grid_pos, y_grid_pos])
    
    def separate_masks(self, multiple_lesions, lesion_idx):
        """
        A function that separates multiple lesion masks into single lesions only 
        """
        # for each lesion: 
        unique_idx = np.unique(multiple_lesions)
        num_lesion_masks = len(unique_idx) - 1
        ds_lesion_mask = multiple_lesions
        row, col, depth = np.shape(ds_lesion_mask)

        img_vol = np.zeros((row, col, depth))

        for i in range(depth):
            img_vol[:,:,i] = (ds_lesion_mask[:,:,i] == lesion_idx)

        return img_vol 

    def _obtain_vol_coords(self):
      
      """
      Given the position of the rectum, make the x,y,z coordinates of the image based on this 

      Returns:
      ---------
      img_coords: list
          List of Meshgrid of image coordinates in x,y,z 
      """
      
      #Initialise coordinates to be 0,0,0 at top left corner of img volume 
      y_vals = np.asarray(range(self.volume_size[0])).astype(float) 
      x_vals = np.asarray(range(self.volume_size[1])).astype(float) 
      z_vals = np.asarray(range(self.volume_size[2])).astype(float) 

      x,y,z = np.meshgrid(x_vals,y_vals, z_vals)

      #Centre coordinates at rectum position
      x-= self.prostate_centroid[0]
      y-= self.prostate_centroid[1]
      z-= self.prostate_centroid[2]

      # Convert to 0.5 x 0.5 x 1mm dimensions 
      img_coords = [x*0.5,y*0.5,z]

      return img_coords

    def _extract_volume_params(self, binary_mask, which_case = 'Tumour'):

      
      """ 
      A function that extracts the parameters of the tumour masks: 
          - Bounding box
          - Centroid
      
      Parameters:
      ------------
      binary_mask : Volume of binary masks
      which_case : string
        Options: 'Tumour' or 'Prostate' 
        Dictates which case we are calculating for : prostate mask or bounding box 

      Returns: 
      ----------
      if which_case == 'Prostate':
      bb_values :  list of 5 x 1 numpy array of bounding box coordinates for Prostate; Radius of sphere for Tumour
      tumour_centroid : ndarray
        centroid coordinates of tumour in x,y,z 

      if which_case == 'Tumour':
      bb_values : float
        Maximum radius from tumour centroid to bounding sphere 
      tumour_centroid:  ndarray
        centroid coordinates of tumour in x,y,z
      """
      
      #Finding pixel indices of non-zero values 
      idx_nonzero = np.where(binary_mask != 0)

      #Min, max values in voxel coordinates in row (y) x col (x) x depth (z)
      min_vals = np.min(idx_nonzero, axis = 1)
      max_vals = np.max(idx_nonzero, axis = 1)
      
      #print(f"Width, height, depth : {max_vals - min_vals}")
      #Obtain bounding box for prostate:
      # tumour centroid is middle of bounding box 
      if which_case == 'Prostate':

        total_area = len(idx_nonzero[0]) #Number of non-zero pixels 
        z_centre = np.round(np.sum(idx_nonzero[2])/total_area)
        y_centre = np.round(np.sum(idx_nonzero[0])/total_area)
        x_centre = np.round(np.sum(idx_nonzero[1])/total_area)
            
        #In pixel coordinates 
        y_dif, x_dif, z_dif = max_vals - min_vals 
        UL_corner = copy.deepcopy(min_vals) #Upper left coordinates of the bounding box 
        LR_corner = copy.deepcopy(max_vals) #Lower right corodinates of bounding box 

        #Centre-ing coordinates on rectum
        UL_corner = UL_corner[[1,0,2]] #- self.rectum_position
        LR_corner = LR_corner[[1,0,2]] #- self.rectum_position

        #Bounding box values : coordinates of upper left corner (closest to slice 0) + width, height, depth 
        bb_values = [UL_corner, x_dif, y_dif, z_dif, LR_corner] 

        tumour_centroid = np.asarray([x_centre, y_centre, z_centre]).astype(int)
        #tumour_centroid2 = ((max_vals + min_vals)/2).astype(int)

        #Using centre of boundineg box as prostat centre : in x,y,z coordinates 
        #tumour_centroid2 = UL_corner + np.array([x_dif/2, y_dif/2, z_dif/2]).astype(int)

      #If tumour: bounding box is bounding sphere; centroid is actual centroid of tumour 
      elif which_case == 'Tumour':
        
        #Extracting centroid tumour 
        total_area = len(idx_nonzero[0]) #Number of non-zero pixels 
        z_centre = np.round(np.sum(idx_nonzero[2])/total_area)
        y_centre = np.round(np.sum(idx_nonzero[0])/total_area)
        x_centre = np.round(np.sum(idx_nonzero[1])/total_area)

        #Calculate tumour centroid 
        tumour_centroid = np.asarray([y_centre, x_centre, z_centre]).astype(int)

        # Find euclidean distance between tumour centroid and list 
        all_coords = np.transpose(np.asarray(idx_nonzero))    #Turn idx non zero into array 
        euclid_dist = cdist(all_coords, np.reshape(tumour_centroid, [1,3])) 
        max_radius = int(np.round(np.max(euclid_dist)))
        #max_radius = int(np.round(np.mean(euclid_dist) + np.std(euclid_dist))) #Use mean instead of max

        #Define bb values as maximum radius 
        bb_values = max_radius 

        #Centre tumour centroid at rectum position : (x,y,z)
        tumour_centroid_centred = np.asarray([x_centre, y_centre, z_centre]) - self.rectum_position
        tumour_centroid = tumour_centroid_centred

      return bb_values, tumour_centroid 

    def find_new_needle_pos(self, action_x, action_y):
      
      """
      A function that computes the relative new x and y positions relative to previous position 
      """

      max_step_size = 5 # changed max step size to 10 
      same_position = False

      x_movement = round_to_05(action_x * max_step_size)
      y_movement = round_to_05(action_y * max_step_size)

      updated_x = self.current_needle_pos[0] + x_movement
      updated_y = self.current_needle_pos[1] + y_movement

      #Dealing with boundary positions 
      x_lower = updated_x < -30
      x_higher = updated_x > 30
      y_lower = updated_y < -30
      y_higher = updated_y > 30

      # Checks if the agent tries to move off the grid at any point
      if x_lower or x_higher or y_lower or y_higher:
        moves_off_grid = True 
      else:
        moves_off_grid = False 

      # Change updated position if agent tries to move out of grid-> stay in the same place or maximum within the grid. 

      if x_lower:
        #Change updated_x to maximum     
        updated_x =  -30

      if x_higher:
        updated_x =  30

      if y_lower: 
        updated_y = -30

      if y_higher: 
        updated_y = 30

      x_grid = updated_x
      y_grid = updated_y 

      new_needle_pos = np.array([int(x_grid), int(y_grid)])

      # Same position if needle_pos_before == new_needle_pos
      if np.all(new_needle_pos == self.current_needle_pos):
        #print("Same needle position")
        same_position = True

      return new_needle_pos, same_position, moves_off_grid

    """ TODO functions """
    
    def add_reg_noise(self, img_vol, tre_ = 3):
        """
        A function that simulates registration noise by adding noise to image coordinates 

        """

        # Add noise to coordinates of prostate volume, tumour volume 


        # TODO - Interpolate volume with noise added to coordinates

        #Image coordinates for interpolation 
        z_ = np.unique(self.img_coords[2])
        y_ = np.unique(self.img_coords[1])
        x_ = np.unique(self.img_coords[0])

        #Add noise to coordinates of the tumour lesion mask 
        noise_array = np.random.normal(loc = 0, scale = np.sqrt((tre_ ** 2) / 3 ), size = (3,))
        tre = np.sqrt(np.sum(noise_array **2))

        #print(noise_array)
        noise_z_ = z_ + noise_array[0]
        noise_y_ = y_ + noise_array[1]
        noise_x_ = x_ + noise_array[2]
        
        tre = np.sqrt(np.sum(noise_array **2))
        x_grid, y_grid, z_grid = np.meshgrid(x_[0::2], y_[0::2], z_[0::4])

        #TODO - come back to adding noise
        interp_array = np.stack([y_grid, x_grid, z_grid], axis = 2) # (y = 0, x = 1, z = 2)
        noise_added_vol = [interpn((noise_y_,noise_x_,noise_z_), img_vol, interp_array[:,:,:,i], bounds_error=False, fill_value=0.0) for i in range(24)]

        noise_added_binarised = np.transpose(noise_added_vol, [1,2,0]) >= 0.5

        return noise_added_binarised, tre

    def create_grid_array(self, x_idx, y_idx, needle_fired, grid_array = None, display_current_pos = False):

      """
      A function that generates grid array coords

      Note: assumes that x_idx, y_idx are in the range (-30,30)

      """

      #Converts range from (-30,30) to image grid array
      x_idx = (x_idx) + 50
      y_idx = (y_idx) + 50

      x_grid_pos = int(x_idx)
      y_grid_pos = int(y_idx)
      
      # At the start of episode no grid array yet   
      first_step = (np.any(grid_array == None)) 

      if first_step:
        grid_array = np.zeros((100,100))
        self.saved_grid = np.zeros((100,100)) # for debugging hit positions on the grid
      else:
        grid_array = self.saved_grid 
    

      # Check if agent has already visited this position; prevents over-write of non-hit pos
      pos_already_hit = (grid_array[y_grid_pos, x_grid_pos] == 1)
      if pos_already_hit:
        value = 1 # Leave the image intensity value as the current value 
      
      else:
        
        #Plot a + for where the needle was fired; 1 if fired, 0.5 otherwise 
        if needle_fired:
            value = 1 
        else:
            value = 0.5
     
      grid_array[y_grid_pos:y_grid_pos+ 2, x_grid_pos] = value
      grid_array[y_grid_pos - 1 :y_grid_pos, x_grid_pos] = value
      grid_array[y_grid_pos, x_grid_pos:x_grid_pos + 2 ] = value
      grid_array[y_grid_pos, x_grid_pos - 1 : x_grid_pos] = value

      # Option to change colour of current grid position 
      if display_current_pos: 
        
        self.saved_grid = copy.deepcopy(grid_array)
        
        # Change colour of current pos to be lower intensity 
        value = 0.25 

        grid_array[y_grid_pos:y_grid_pos+ 2, x_grid_pos] = value
        grid_array[y_grid_pos - 1 :y_grid_pos, x_grid_pos] = value
        grid_array[y_grid_pos, x_grid_pos:x_grid_pos + 2 ] = value
        grid_array[y_grid_pos, x_grid_pos - 1 : x_grid_pos] = value

        # in the next iteration, saved grid will be current grid with no marked black value 



      return grid_array

    def check_needle_hits_outside_prostate(self, needle_traj):
      """
      A function that checks if the needles hit the prostate 

      Returns true if needle hits OUTSIDE prostate, otherwise false 
      """

      intersection_volume = needle_traj * self.img_data['prostate_mask']
    
      if np.all(intersection_volume == 0):
        # no intersection, therefore hits OUTSIDE the prostate 
        return True

      else:
        # intersects with prostate mask, so hits prostate 
        return False 
      
    def get_lesion_centroid(self):
      """
      A function that returns lesion centroid of lesion idx sampled during training
      """
      
      lesion_centroid = self.tumour_centroids[self.lesion_idx] + self.prostate_centroid
      #print(self.tumour_centroids)
      #print(f"lesion idx {self.lesion_idx}")
      return lesion_centroid, self.true_tumour_vol
    
    def get_patient_name(self):
      """
      A function that returns patient naem for IL files 
      """
      
      return self.patient_name[0]
    
    def get_initial_obs(self):
      return self.initial_obs
    
    def get_single_lesion_mask(self):
      return self.true_tumour_vol # single lesion mask 
    
    def get_prostate_centroid(self):
      return self.prostate_centroid
    
    def get_needle_depth(self):
      """
      Prints whether aopex or base needle shoudl be fired 
      
      apex if lesion <= max_needle_depth/2 
      base otherwise 
      """
      max_lesion_depth = np.max(np.where(self.true_tumour_vol == 1)[-1])
      
      if max_lesion_depth <= (self.max_needle_depth/2):
        # apex
        return 1
      else: 
        # base 
        return 2 
    
    def compute_reward_simple_reward(self, done, ccl_coeff, hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid, scale_factor = 100):
        
      """
      Computes simple reward function 
      
      Parameters:
      done: bool array to signify whether episode is done for final reward 
      same_position: signify whether agent stays at same position 
      magnitude: scalar value to scale the corrleation coeffieicnt by 
      
      Notes:
      --------
      1. CCL: ccl_coefficienct from previous experiences 
      2. Hit_rate : number_lesions_hit  / num_lesions_present
      3. Efficiency : number_needles_hit / num_needles_fired

      """
      if done:

        if np.isnan(ccl_coeff):
          ccl_coeff = 0 
        
        #print(f'CCL COEFF {ccl_coeff}')

        #BONUS reward of 10 * hit_rate (max 10, minimum 0 )
        min_hit_threshold = 0.5 
        if hit_rate > min_hit_threshold: 
          reward = scale_factor
        else: 
          reward = -50 # did not hit the minimum number of lesions 
        reward = hit_rate * scale_factor

        #Large penalty of exceeding number of needles fired 
        if max_num_needles_fired:
          reward -= 50 

      else:
        
        if needle_fired: 
            if needle_hit:
                reward = 10 
            else:
                reward = -1 
        
        # No needles fired, but still taking up time navigating 
        else:
            reward = -0.8 

      # Penalise for staying at the same position or moving off the grid
      if same_position: 
          reward -= 5 

      if moves_off_grid:
          reward -= 10

      return reward

    def compute_reward(self, needle_fired, needle_hit, outside_prostate):
        
      """
      Computes simple reward function 
      
      Parameters:
      done: bool array to signify whether episode is done for final reward 
      same_position: signify whether agent stays at same position 
      magnitude: scalar value to scale the corrleation coeffieicnt by 
      
      """

      reward = 0 

      # if fired : +10 for lesion 
      if needle_fired: 

        if needle_hit: 
          reward += 10
        else:
          reward -= 2 # for missing 

      # if not fired, -1 for taking up time 
      else:
        reward -= 1 # for taking up time and not firing 

      if outside_prostate:
        reward -= 5
    
      return reward

    def compute_reward_classify(self, needle_fired, needle_hit, outside_prostate, label_z, pred_z):
        
      """
      Computes reward function with classificatoin 
      
      Parameters:
      done: bool array to signify whether episode is done for final reward 
      same_position: signify whether agent stays at same position 
      magnitude: scalar value to scale the corrleation coeffieicnt by 
      
      """

      # navigation reward : x,y 
      nav_reward = 0 
       
      # TODO : consider needle hit AND needle fired; if not hit then no penalty 
      if needle_hit: 
        nav_reward += 10 
      else : 
        nav_reward -= 2
      
      # hit reward : z depth ; classification reward 
      hit_reward = 0 

      if label_z == pred_z: 
        hit_reward += 10 
      else: 
        hit_reward = 0 

      reward = (2/3)*nav_reward + (1/3)*hit_reward  # z reward + x,y reward; z reward is 1/3 weighted 

      if outside_prostate:
        reward -= 3
    
      return reward

    def compute_reward_penalty(self, needle_fired, needle_hit, outside_prostate, label_z, pred_z, done):
        
      """
      Computes reward function with classificatoin 
      
      Parameters:
      done: bool array to signify whether episode is done for final reward 
      same_position: signify whether agent stays at same position 
      magnitude: scalar value to scale the corrleation coeffieicnt by 
      
      """
      reward = 0 
      
      if done:

        # Add addiitionl reward for episode being done, but no needles are fired 
        if self.num_needles <= 2: # less than 3 needles fired 
          reward = -100 # huge penalty for not firing enough enedles 
        
      else:
        
        ### navigation reward : x,y 
        nav_reward = 0 
        
        # TODO : consider needle hit AND needle fired; if not hit then no penalty 
        if needle_hit: 
          # Penalise for firing more than 5 times 
          nav_reward += 10
            
        else : 
          nav_reward -= 2
        
        ### hit reward : z depth ; classification reward 
        hit_reward = 0 

        if label_z == pred_z:
          #print(f"Label z : {label_z} pred_z : {pred_z}") 
          
          if pred_z == 0: 
            hit_reward = 0 #reward for predicting not to fire : no reward for this!!! 
            
          else: #reward for predicting to +fire correctly apex or base position more 
            hit_reward += 10 # super reward for predicting apex / base firing 
        
        else: 
          # if label_z == 0: # not meant to fire 
          #   hit_reward = -1
          
          # else: #if could fire, but chose not to, no penalty 
          #  hit_reward = 0  
          hit_reward = -1 # penalty for mis-firing (ie firing when predicted z = 0 when z = 1/2) or predicitng z = 1 when z = 0
        
        reward = (2/3)*nav_reward + (1/3)*hit_reward  # z reward + x,y reward; z reward is 1/3 weighted 

        # for debugging: 
        if needle_hit: 
          print(f"Needle fired and hit; hit reward : {hit_reward} nav reward : {nav_reward} total : {reward}")
        
        #else:
          #print(f"Needle fired but not hit; Hit reward : {hit_reward} nav reward : {nav_reward} total : {reward}")

        if outside_prostate:
          reward -= 5 # increase penalty for firing outside prostate
      
      return reward
    
    def compute_reward_metrics(self, hr, needle_fired, needle_hit, outside_prostate, done, metric = 'ccl'):
        
      """
      Computes reward function based on CCL norm, hit_rate 
      
      Parameters:
      done: bool array to signify whether episode is done for final reward 
      same_position: signify whether agent stays at same position 
      magnitude: scalar value to scale the corrleation coeffieicnt by 
      
      """
      
      # reward is average ccl_norm : ie np.mean(ccl_norm)
      
      # ccl is obtained after needle sampling is accomplished! 
      if done: 
        
        reward = 0 
        
        if metric == 'ccl':
          ccl_norm = np.mean(self.all_norm_ccl)
          reward = ccl_norm * 100 # percentage of ccl norm 
        
        elif metric == 'hr':
          reward = hr * 100 # percentage of ccl norm 

        elif metric == 'coverage':
          #TODO
          raise("Not yet implemented")
        
        else: 
          # combination of all of them 
          print(f"Metrics : Combined")
          reward = 100*((ccl_norm)*0.5 + hr*0.5) #equally weight ccl norm and hr 
        
        print(f"Done : reward : {reward}")
      else: 
        # Reward shaping : 
        # Small intermediate rewards
        reward = 0 

        # if fired : +10 for lesion 
        if needle_fired: 

          if needle_hit: 
            reward += 1
          else:
            reward -= 0.2 # for missing 

        # if not fired, -1 for taking up time 
        else:
          reward -= 0.1 # for taking up time and not firing 

        if outside_prostate:
          reward -= 0.5
          
      return reward


    def compute_needle_traj(self, x_grid, y_grid, depth, noise_added = False):  
        """

        A function that computes the needle trajectory mask, to use for computing the CCL

        x_grid : ndarray 
        y_grid : ndarray 
        Noise_added : bool whether or not to add noise to needle trajectory

        Notes:
        ---------
        x_grid and y_grid are in image coordinates, need to be changed from grid coordinates to image 

        """
        
        if noise_added: 
            # TODO - add noise to needle trajectory 
            pass
          
        # Change x_grid, y_grid from lesion coords to image coords; centre around prostate centroid 
        x_pos = (x_grid)*2 + self.prostate_centroid[0] # x_grid and y_grid multiplied by 2 to account for 0.5x0.5x1 resolution of mri dataset
        y_pos = (y_grid)*2 + self.prostate_centroid[1]
        
        depth_map = {1 : 0.5, 2 : 1}
        
        # 16g/18g corresponds to 1.2mm, 1.6mm diameter ie 3 pixels taken up on x,y plane 
        needle_mask = np.zeros_like(self.img_data['mri_vol'])
        
        if depth != 0:
          scale_factor = depth_map[depth]
          needle_mask[y_pos -1 : y_pos +2 , x_pos -1 : x_pos +2, 0:int(scale_factor *self.max_needle_depth)] = 1

        # needle_mask = np.zeros_like(self.img_data['mri_vol'])
        # for x_grid in range(-30,35,5):
        #   for y_grid in range(-30,35,5):
            
        #       x_pos = (x_grid)*2 + self.prostate_centroid[0] # x_grid and y_grid multiplied by 2 to account for 0.5x0.5x1 resolution of mri dataset
        #       y_pos = (y_grid)*2 + self.prostate_centroid[1]
        
        #       needle_mask[y_pos -1 : y_pos +2 , x_pos -1 : x_pos +2, 0:int(scale_factor *self.max_needle_depth)] = 1


        # compute intersection volume 
        intersection_volume = needle_mask * self.true_tumour_vol

        # Computer intersection voluem for reward 
        virtual_needle_mask = np.zeros_like(self.img_data['mri_vol'])
        virtual_needle_mask[y_pos -1 : y_pos +2 , x_pos -1 : x_pos +2, 0:self.max_needle_depth] = 1 

        # compute intersection volume for reward computation 
        intersection_reward = virtual_needle_mask * self.true_tumour_vol 

        if len(np.unique(intersection_reward)) != 1: # if only 0 

          z_tumour = np.max(np.where(intersection_reward)[-1])

          # apex if tumour depth < halfway of prostate gland depth else base 
          apex = (z_tumour <= (self.max_needle_depth/2))
          
          if apex: 
            ground_truth_z = 1 
          else:
            ground_truth_z = 2 

        else: 
          ground_truth_z = 0 

        return needle_mask, intersection_volume, ground_truth_z 

    def compute_ccl(self, intersection_volume):

      """
      Computes the CCL given the intersection volume

      Parameters:
      ------------
      intersection_volume : 200 x 200 x 96 array of intersection 

      Returns:
      -----------
      ccl : end point - begin point from tip to end of needle euclidean distance 
      ccl_approximate : simply computes differnece in z as an approximate to ccl 
      """
      
      # ie binary vol only contains 0s 
      no_intersect = (len(np.unique(intersection_volume)) == 1)
     
      # compute max ccl (approximate only! from bb of tumour)
      all_z_tum = (np.where(self.true_tumour_vol == 1))[-1]
      max_ccl = np.max(all_z_tum) - np.min(all_z_tum)
      
      if no_intersect:
        ccl = 0 
        ccl_approximate = 0 
      
      else:
        y_vals, x_vals, z_vals = np.where(intersection_volume == 1)
        idx_max = np.where(z_vals == np.max(z_vals))[0] #idx where coords are maximum z depth 
        idx_min = np.where(z_vals == np.min(z_vals))[0]

        # Compute average centres at z_min and z_max, and compute euclidean distance   
        begin_point = np.array([np.mean(x_vals[idx_min]), np.mean(y_vals[idx_min]), np.min(z_vals)])
        end_point = np.array([np.mean(x_vals[idx_max]), np.mean(y_vals[idx_max]), np.max(z_vals)])

        ccl_approximate = np.max(z_vals) - np.min(z_vals) #can be used as ccl if no noise is added to needle_traj 
        ccl = np.sqrt(np.sum((end_point - begin_point) ** 2))


      return ccl, ccl_approximate, max_ccl 

    def compute_ccl_multiple_lesions(self, needle_mask):
        """
        Computes CCL given needle mask and tumour masks when multiple lesions are present! 

        Params:
        needle_mask : ndarray 200 x 200 x 96 binary mask of needle trajectory 

        Notes:
        ------
        This function could fail if two lesions are behind each other. Needle could interesect both at the same time
        Need to check for over-lapping lesions! 

        Assumes that only one lesion is hit at the same time 

        """

        #inner function to compute needle traj from intersection volume 
        def compute_ccl_given_idx(intersection_volume, lesion_idx):
            # Compute needle values 
            y_vals, x_vals, z_vals = np.where(intersection_volume == lesion_idx)
            idx_max = np.where(z_vals == np.max(z_vals))[0] #idx where coords are maximum z depth 
            idx_min = np.where(z_vals == np.min(z_vals))[0]

            # Compute average centres at z_min and z_max, and compute euclidean distance   
            begin_point = np.array([np.mean(x_vals[idx_min]), np.mean(y_vals[idx_min]), np.min(z_vals)])
            end_point = np.array([np.mean(x_vals[idx_max]), np.mean(y_vals[idx_max]), np.max(z_vals)])

            ccl_approximate = np.max(z_vals) - np.min(z_vals) #can be used as ccl if no noise is added to needle_traj 
            ccl = np.sqrt(np.sum((end_point - begin_point) ** 2))

            return ccl, ccl_approximate
          
        intersection_volume = needle_mask * self.multiple_label_img
        unique_idx = np.unique(intersection_volume)

        #ie no intersection between needle and lesion mask
        if len(unique_idx) == 1: 
            #no ccl and no idx hit
            #print(f"ccl = 0 no intersection between lesion and mask")
            return 0, None

        #ie needle hits multiple lesions (behind each other) --> need to obtain separate CCL for both 
        if len(unique_idx) > 2:
            # Compute CCL separately for each needle trajectory
            #print("Multiple lesions hit by needle")

            ccl_vals = []
            lesion_idx_hit = [] 

            for idx in unique_idx[1:]:

              #Compute CCL separately 
              ccl, ccl_approx = compute_ccl_given_idx(intersection_volume, idx)
              ccl_vals.append(ccl)

              #Remove 1 as background is 1
              lesion_idx_hit.append(int(idx) - 1)
            
            ccl = ccl_vals

        #one lesion hit 
        else: 
            #print("one lesion hit")
            lesion_idx_hit = int(unique_idx[1]) - 1 # 0 is background so remove -1 to account for lesion number
            ccl, ccl_approx = compute_ccl_given_idx(intersection_volume, unique_idx[1])

        return ccl, lesion_idx_hit 

    def compute_max_ccl(self, tumour_mask):
      
      """
      Computes maximum ccl for reward function
      """
      img_shape = np.shape(tumour_mask)
      x_coords = np.arange(0, img_shape[0])
      y_coords = np.arange(0, img_shape[1])
      z_coords = np.arange(0, img_shape[2])
      x,y,z = np.meshgrid(x_coords,y_coords, z_coords)
      
      z_mask = z * tumour_mask
      z_mask[z_mask == 0] = float("nan")
      np.seterr(divide='ignore')
      z_min = np.nanmin(z_mask, axis = 2)
      z_max = np.nanmax(z_mask, axis = 2)
      max_ccl = np.nanmax(z_max - z_min)

      return max_ccl 

    def apply_deforms(self, volume, rate = 0.25, scale = 0.1, threshold = 0.45):
      """      
      A function that deforms the prostate and lesion glands together

      Parameters:
      :volume: ndarray 2 x 100 x 100 x 25
      """
      
      # Generate new transform per time step 
      self.random_transform.generate_random_transform(rate = rate, scale = scale)
      
      # Apply same transformation to lesion and prostate gland 
      deformed_vol = self.random_transform.warp(volume.type(torch.float32))
      deformed_vol = 1.0*(deformed_vol >= threshold)
      
      return deformed_vol
    
if __name__ == '__main__':

    #Evaluating agent on training and testing data 
    #ps_path = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality'
    #rectum_path = '/Users/ianijirahmae/Documents/PhD_project/rectum_pos.csv'
    #csv_path = '/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv'

    ps_path = '/raid/candi/Iani/MRes_project/Reinforcement Learning/DATASETS/'
    csv_path = '/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv'
    #rectum_path = '/raid/candi/Iani/MRes_project/Reinforcement Learning/rectum_pos.csv'
    
    log_dir = 'test'
    os.makedirs(log_dir, exist_ok=True)

    PS_dataset_train = Image_dataloader(ps_path, csv_path, use_all = False, mode  = 'train')
    Data_sampler_train = DataSampler(PS_dataset_train)

    Biopsy_env_init = TemplateGuidedBiopsy_single(Data_sampler_train, results_dir = log_dir, max_num_steps = 20, reward_fn = 'patient', obs_space = 'both', start_centre = False) #Data_sampler_train,
    
    test_obs = Biopsy_env_init.reset()
    new_obs = Biopsy_env_init.step([-1.5,1,1])

    # Initiate random policy 


    print('chicken')

    def evaluate(model, num_episodes=100, deterministic=True):
      """
      Evaluate a RL agent
      :param model: (BaseRLModel object) the RL Agent
      :param num_episodes: (int) number of episodes to evaluate it
      :return: (float) Mean reward for the last num_episodes
      """
      # This function will only work for a single Environment
      vec_env = model.get_env()
      all_episode_rewards = []
      for i in range(num_episodes):
          episode_rewards = []
          done = False
          obs = vec_env.reset()
          num_steps = 0 
          while not done:
              # _states are only useful when using LSTM policies
              action, _states = model.predict(obs, deterministic = False)
              # here, action, rewards and dones are arrays
              # because we are using vectorized env
              # also note that the step only returns a 4-tuple, as the env that is returned
              # by model.get_env() is an sb3 vecenv that wraps the >v0.26 API
              obs, reward, done, info = vec_env.step(action)
              print(f"Actions: {action} reward : {reward}")
              episode_rewards.append(reward)
              all_episode_rewards.append(sum(episode_rewards))
              num_steps += 1 

          print(f"Done after: {num_steps} cummulative reward : {sum(episode_rewards)}")

      mean_episode_reward = np.mean(all_episode_rewards)
      std_reward = np.std(all_episode_rewards)
      print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

      return mean_episode_reward, std_reward
    
    # initialise agent
    policy_kwargs = dict(features_extractor_class = NewFeatureExtractor, features_extractor_kwargs=dict(multiple_frames = True))
    agent = PPO(CnnPolicy, Biopsy_env_init, policy_kwargs = policy_kwargs, n_epochs = 3, learning_rate = 0.0001, tensorboard_log = 'test')
    #model = PPO(CnnPolicy, Biopsy_env_init, verbose=0)

    mean_reward, std_reward = evaluate(agent, num_episodes = 10)

    print('chicken')