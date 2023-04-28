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

from rl_utils import *
#Import all dataloader functions 
from Prostate_dataloader import * 
from stable_baselines3.ppo.policies import CnnPolicy#, MultiInputPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
import torch 
import time 

from PIL import Image, ImageDraw, ImageFont

import matplotlib
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

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

class TemplateGuidedBiopsy_penalty(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, DataSampler, obs_space = 'images', results_dir = 'test', env_num = '1', reward_fn = 'penalty', \
    miss_penalty = 2, terminating_condition = 'max_num_steps', train_mode = 'train', device = 'cpu', max_num_steps = 100, \
      penalty = 5, reward_magnitudes = [1/3, 1/3, 1/3], start_centre = False, inc_HR = True, inc_CCL = False):

        """
        Actions : delta_x, delta_y, z (fire or no fire or variable depth)
        """
        super(TemplateGuidedBiopsy_penalty, self).__init__()

        self.obs_space = obs_space

        ## Defining action and observation spaces
        self.action_space = spaces.Box(low = -1, high= 1, shape = (3,), dtype = np.float32)

        #if obs_space == 'images':
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(100, 100, 25), dtype=np.float64)

        # Load dataset sampler 
        self.DataSampler = DataSampler
        
        # starting condition on grid (within centre of template grid)
        img_data = self.sample_new_data()

        # Defining variables 
        self.done = False 
        self.reward_fn = reward_fn
        self.num_needles = 0 
        self.max_num_needles = 4 * self.num_lesions
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
        self.reward_magnitudes = reward_magnitudes
        self.penalty_reward = penalty 
        self.terminating_condition = terminating_condition
        self.needle_penalty = miss_penalty
        self.start_centre = start_centre

        # Reward including factors
        self.inc_HR = inc_HR
        self.inc_CCL = inc_CCL 

        # Initialise state
        initial_obs, starting_pos = self.initialise_state(img_data, start_centre)

        # Save starting pos for next action movement 
        self.current_needle_pos = starting_pos 

        # If not just using images, concatenate num needles left to obs space 
        #if self.obs_space != 'images':
        #  initial_obs = {'img_volume': initial_obs, 'num_needles_left' : self.max_num_needles - self.num_needles}
        
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
            self.num_data = 105 #402
        elif train_mode == 'test':
            self.num_data = 30 #115
        else:
            self.num_data = 15 #58

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

        # 1. Convert z from fire, no fire 
        fire_prob = (action[2] + 1) / 2 # convert from (-1,1) to (0,1)
        #needle_fired = fire_prob >= 0.5 
        needle_fired = action[2] > 0

        # 2. Move current template pos according to action_x, action_y -> DOUBLE CHECK THIS 
        grid_pos, same_position, moves_off_grid = self.find_new_needle_pos(action[0], action[1])
        self.current_needle_pos = grid_pos 

        # 3. Update state, concatenate grid_pos to image volumes 
        new_grid_array = self.create_grid_array(grid_pos[0], grid_pos[1], needle_fired, self.grid_array, display_current_pos = True)

        # 4. Compute CCL if needle fired and append to list of CCL_coeff
        needle_hit = False 

        # 4. Obtain needle sample trajectory, compute CCL using ground truth masks
        needle_traj = self.compute_needle_traj(grid_pos[0], grid_pos[1]) #note grid_pos[0] and grid_pos[1] need to correspond to image coords
        ccl, lesion_idx = self.compute_ccl(needle_traj)

        # For computing if a new lesion was hit by needle -> exploration reward!!! 
        if lesion_idx != None:
          # np.any is used to account for when multiple lesions are hit!! 
          new_lesion_hit = np.any((self.num_needles_per_lesion[lesion_idx] == 0)) 

        # None means no lesions were hit, so new lesions hit 
        else:
          new_lesion_hit = False 

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



            # If two lesions were hit by the same needle, append each ccl and size separately
            two_lesions_hit = (type(ccl) == list)

            if two_lesions_hit:
              
              for i in range(len(ccl)):
                self.all_ccl.append(ccl[i])
                self.all_sizes.append(self.tumour_statistics['lesion_size'][lesion_idx[i]])
                self.num_needles_per_lesion[lesion_idx[i]] += 1
              
              #Increase successful needle count
              self.num_needles_hit +=1 
              needle_hit = True 

            # If one or no lesions were hit by needle 
            else: 

              self.all_ccl.append(ccl)
            
              # No lesion therefore no lesion size 
              if lesion_idx == None:
                  self.all_sizes.append(0)

              # Single lesion 
              else:
                  self.all_sizes.append(self.tumour_statistics['lesion_size'][lesion_idx])
                  self.num_needles_per_lesion[lesion_idx] += 1
                  
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

            #Use previous ccl correlation as no update to ccl values 
            ccl_corr_online = self.previous_ccl_online   
            needle_hits_outside_prostate = False 

            needle_traj = np.zeros_like(self.img_data['mri_vol'])
        
        # Add number of needles left as additional info 

        needles_left = self.max_num_needles - self.num_needles
        new_grid_array = add_num_needles_left(new_grid_array, needles_left)
          
        #new_obs = self.obtain_obs(new_grid_array)
        new_obs = self.obtain_obs_needle(new_grid_array, needle_traj)

        # 5. Check if episode terminates episode if hit rate threshold is reached or max_num_steps is reached 
        
        # Commpute statistics 
        all_lesions_hit = np.all(self.num_needles_per_lesion >= 1)
        agent_hit_rate = np.mean((self.num_needles_per_lesion >= 2)) # how many lesions are hit at least twice 
        hit_threshold_reached = agent_hit_rate >= self.hit_rate_threshold # if hit rate threshold is reached 
        max_num_needles_fired = (self.num_needles >= self.max_num_needles) 
        max_num_steps_reached = (self.step_count >= self.max_num_steps_terminal) 
        #max_num_steps_reached = (self.step_count >= (self.max_num_needles + 10)) 

        # Compute efficiency 
        if self.num_needles == 0:
          efficiency = 0 
        else:
          efficiency = self.num_needles_hit / self.num_needles

        
        # Terminate depending on whether max num steps are reached OR if hit threshold is reached 
        if self.terminating_condition == 'max_num_steps':
          terminate = max_num_steps_reached 
        elif self.terminating_condition == 'hit_threshold':
          terminate = max_num_steps_reached or hit_threshold_reached 
        elif self.terminating_condition == 'max_num_needles_fired':
          terminate = max_num_steps_reached or max_num_needles_fired

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
        if self.reward_fn == 'patient':
            reward =  self.compute_reward_ccl_sum(done_new_patient, ccl_corr_online, agent_hit_rate, efficiency, same_position = same_position, moves_off_grid = moves_off_grid, magnitude = self.reward_magnitudes)
        elif self.reward_fn == 'patient_b':
            reward =  self.compute_reward_ccl_multiply(done_new_patient, ccl_corr_online, agent_hit_rate, self.num_needles, same_position = same_position, moves_off_grid = moves_off_grid)
        elif self.reward_fn == 'penalty':
            reward = self.compute_reward_ccl_penalty(done_new_patient, ccl_corr_online, agent_hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid)
        elif self.reward_fn == 'reward':
            reward = self.compute_reward_ccl_reward(done_new_patient, ccl_corr_online, agent_hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid)
        elif self.reward_fn == 'reward_smallpenalty':
            reward = self.compute_reward_ccl_reward_smallpenalty(done_new_patient, ccl_corr_online, agent_hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid, penalty = self.penalty_reward)
        elif self.reward_fn == 'reward_hit_rate':
            reward = self.compute_reward_hit_rate(done_new_patient, ccl_corr_online, agent_hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid, miss_penalty = self.needle_penalty , penalty = self.penalty_reward)
        elif self.reward_fn == 'ccl_coeff':
            reward = self.compute_reward_ccl(done_new_patient, ccl_corr_online, agent_hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid, miss_penalty = self.needle_penalty , penalty = self.penalty_reward)
        elif self.reward_fn == 'simple':
          reward = self.compute_reward_simple_reward(done_new_patient, ccl_corr_online, agent_hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid, inc_HR = self.inc_HR, inc_CCL = self.inc_CCL)
        elif self.reward_fn == 'ccl_only':
          reward = self.compute_reward_ccl_only(done_new_patient, ccl_corr_online, agent_hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid, inc_HR = self.inc_HR, inc_CCL = self.inc_CCL)
        elif self.reward_fn == 'BASIC':
          reward = self.compute_reward_BASIC(needle_fired, needle_hit, penalty_reward = 10)
        elif self.reward_fn == 'BASIC_20':
          reward = self.compute_reward_BASIC_20(needle_fired, needle_hit, penalty_reward = 10)
        elif self.reward_fn == 'BASIC_FIRED':
          reward = self.compute_reward_BASIC_FIRED(needle_fired, needle_hit, done_new_patient, penalty_reward = 10)



        ### Additional rewards : 

        # Check if needle hits OUTSIDE prostate, apply additional penalty of -5 
        prostate_penalty = -10
        
        if needle_hits_outside_prostate:
          reward += prostate_penalty 

        # Check if new lesion explored / new area explored -> add bonus reward!!!
        #lesion_exploration_reward = 10
        #if new_lesion_hit: 
        #  reward += lesion_exploration_reward 
        
        # Save ccl_corr as new previous_ccl_corr
        self.previous_ccl_corr = ccl_corr_online

        # 6. Compute saving statistics and actions print(f"Reward : {reward}")
        saved_actions = np.array([grid_pos[0], grid_pos[1], self.max_needle_depth*int(needle_fired), reward])

        with open(self.file_output_actions, 'a') as fp:
          np.savetxt(fp, np.reshape(saved_actions, [1,-1]), '%s', ',')

        info = {'num_needles_per_lesion' : self.num_needles_per_lesion, 'all_ccl' : self.all_ccl,\
             'all_lesion_size' : self.all_sizes, 'all_lesions_hit' : all_lesions_hit,
             'ccl_corr' : ccl_corr_online, 'hit_rate' : agent_hit_rate, \
               'new_patient' : done_new_patient, 'ccl_corr_online' : ccl_corr_online, 'efficiency' : efficiency,
                'num_needles' : self.num_needles, 'max_num_needles' : self.max_num_needles, 'num_needles_hit' : self.num_needles_hit, 'firing_grid' : self.firing_grid}

        # Reset ccl statistics after going through entire dataset ie zero ccl and all_lesion_sizes
        if self.patient_counter  >= self.num_data:
            self.reset_ccl_statistics()
            self.all_ccl = [] 
            self.all_sizes = [] 
            self.patient_counter = 0 
            bonus_reward = 100 * ccl_corr_online
            reward += bonus_reward 

        return new_obs, reward, done_new_patient, info

    def reset_ccl_statistics(self):
        
        img_data = self.sample_new_data()
        initial_obs, starting_pos = self.initialise_state(img_data, self.start_centre)
        #print(f"Starting pos : {starting_pos}")

        # Save starting pos for next action movement 
        self.current_needle_pos = starting_pos 

        # Defining variables 
        self.done = False 
        self.num_needles = 0 
        self.num_needles_hit= 0 
        self.max_num_steps = 4 * self.num_lesions
        self.num_needles_per_lesion = np.zeros(self.num_lesions)
        self.all_ccl = [] 
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
        initial_obs, starting_pos = self.initialise_state(img_data, self.start_centre)

        #if self.obs_space != 'images':
        #  initial_obs = {'img_volume': initial_obs, 'num_needles_left' : self.max_num_needles - self.num_needles}
        #print(f"Starting pos : {starting_pos}")

        # Save starting pos for next action movement 
        self.current_needle_pos = starting_pos 

        # Defining variables 
        self.done = False 
        self.num_needles = 0 
        self.num_needles_hit = 0 
        self.max_num_steps = 4 * self.num_lesions
        self.max_num_needles = 4 * self.num_lesions
        self.num_needles_per_lesion = np.zeros(self.num_lesions)
        #self.all_ccl = [] 
        #self.all_sizes = [] 
        self.step_count = 0 
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
    

    def obtain_obs_needle(self, template_grid, needle_mask):
        
        """
        Obtains observations from current template grid array and stacks them 

        Notes:
        ----------
        Down-samples and only obtains every 2 pixels for CNN efficiency 

        """

        needle_vol = needle_mask[0::2, 0::2, 0::4]
        prostate_vol = self.noisy_prostate_vol[:, :, :] 
        tumour_vol = self.noisy_tumour_vol[:, :, :]
        combined_vol = np.zeros_like(needle_vol)

        # Prostate : 0.25, Needle : 0.5, tumour : 0.75, tumour/needle : 1 
        combined_vol[prostate_vol == 1] = 0.25
        combined_vol[needle_mask[0::2, 0::2, 0::4] == 1] = 0.5 
        combined_vol[tumour_vol == 1] = 0.75
        combined_vol[(tumour_vol + needle_vol) == 2] = 1 

        new_obs = np.concatenate([np.expand_dims(template_grid, axis = 2), combined_vol], axis = 2)
        
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
        self.max_needle_depth = np.max(np.where(self.img_data['prostate_mask'] == 1)[-1]) #max z depth with prostate present

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
        :state: 200 x 200 x 193 dimensions x 4
        """

        # 1. Initialise starting point on template grid : within centre box of grid 
        all_possible_points  = np.arange(-15, 20, 5)

        if start_centre: 
          starting_x = 0
          starting_y = 0 
        else:
          starting_x, starting_y = np.random.choice(all_possible_points, 2)

        # 2. Obtain grid of initial needle starting position 
        grid_array = self.create_grid_array(starting_x, starting_y, needle_fired = False, display_current_pos = True)
        self.grid_array = copy.deepcopy(grid_array)

        # Include number of needles left as additional info on image 
        grid_array = add_num_needles_left(grid_array, self.max_num_needles)
        
        starting_points = np.array([starting_x, starting_y]) 

        # 3. Obtain noisy prostate and tumour masks (add reg noise )
        self.noisy_prostate_vol, tre_prostate = self.add_reg_noise(img_vol['prostate_mask'], tre_ = 3)
        self.noisy_tumour_vol, tre_tumour = self.add_reg_noise(img_vol['tumour_mask'], tre_ = 4) # more tre for tumour

        #from matplotlib import pyplot as plt
        #fig, axes = plt.subplots(2)
        #axes[0].imshow(self.noisy_tumour_vol[:,:,22])
        #axes[1].imshow(img_vol['tumour_mask'][0::2,0::2,0::2][:,:,22])
        obs = self.obtain_obs_needle(grid_array, np.zeros_like(self.img_data['mri_vol']))

        return obs, starting_points 

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

      max_step_size = 10
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

      new_needle_pos = np.array([x_grid, y_grid])

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
        
    def compute_reward_ccl_sum(self, done, ccl_coeff, hit_rate, efficiency, same_position, moves_off_grid, magnitude = [1/3, 1/3, 1/3], scale_factor = 100):
        
        """
        Computes reward function given CCL and if needle is fired or not 
        
        Parameters:
        done: bool array to signify whether episode is done for final reward 
        same_position: signify whether agent stays at same position 
        magnitude: scalar value to scale the corrleation coeffieicnt by 
        
        Notes:
        --------
        1. CCL: ccl_coefficienct from previous experiences 
        2. Hit_rate : number_lesions_hit / num_lesions_present
        3. Efficiency : number_needles_hit / num_needles_fired

        """
        
        if done:

          #figure_plot = plt.figure()
          #plt.scatter(self.all_sizes , self.all_ccl)
          #plt.xlabel("Lesion sizes (number of voxels)")
          #plt.ylabel("CCL (mm)")
          #self.current_ccl_plot = plt.gcf()
          #plt.close()

          if np.isnan(ccl_coeff):
            ccl_coeff = 0 
          
          print(f'CCL coeff: {ccl_coeff}')

          ccl_reward = magnitude[0] * ccl_coeff 
          hit_reward = magnitude[1] * hit_rate
          eff_reward = magnitude[2] * efficiency

          # Reward should sum to maximum of 1 * scale_factor (in this case 10 is chosen)
          reward = scale_factor * (ccl_reward + hit_reward + eff_reward)
          #print(f'Reward {reward} ccl_coeff : {ccl_coeff} hit rate : {hit_rate} num_needles_fired : {efficiency}')

        else:
          reward = 0

        # Penalise for staying at the same position or moving off the grid
        if same_position: 
            reward -= 1 

        if moves_off_grid:
            reward -= 1

        return reward
    
    def compute_reward_ccl_multiply(self, done, ccl_coeff, hit_rate, num_needles_fired, same_position, moves_off_grid, scale_factor = 100):
        
      """
      Computes reward function given CCL and if needle is fired or not 
      
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

        # Reward should sum to maximum of 1 * scale_factor (in this case 10 is chosen)
        if np.isnan(ccl_coeff):
          ccl_coeff = 0 
        
        reward = (ccl_coeff * hit_rate * scale_factor) / num_needles_fired 
        #print(f'Reward {reward} ccl_coeff : {ccl_coeff} hit rate : {hit_rate} num_needles_fired : {num_needles_fired}')

      else:
        reward = 0

      # Penalise for staying at the same position or moving off the grid
      if same_position: 
          reward -= 1 

      if moves_off_grid:
          reward -= 1

      return reward
    
    def compute_reward_ccl_penalty(self, done, ccl_coeff, hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid, scale_factor = 100):
        
      """
      Computes reward function with added penalty for exceeding num needles fired 
      
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

        # Reward should sum to maximum of 1 * scale_factor (in this case 10 is chosen)
        if np.isnan(ccl_coeff):
          ccl_coeff = 0 
        
        print(f'CCL COEFF {ccl_coeff}')
        reward = (ccl_coeff * hit_rate * scale_factor) 
        #print(f'Reward {reward} ccl_coeff : {ccl_coeff} hit rate : {hit_rate} num_needles_fired : {num_needles_fired}')

      else:
        
        if needle_fired: 
            if needle_hit:
                reward = 1 
            else:
                reward = -1 
            
            # Check if fired needle traj hits the prostate 

        # No needles fired, but still taking up time navigating 
        else:
            reward = -0.5 
    
      # Penalty for firing more needles than max needles fired of -10 
      if max_num_needles_fired:
          reward -= 10 

      # Penalise for staying at the same position or moving off the grid
      if same_position: 
          reward -= 1 

      if moves_off_grid:
          reward -= 1

      return reward

    def compute_reward_ccl_reward_smallpenalty(self, done, ccl_coeff, hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid, penalty = 5, scale_factor = 100):
        
      """
      Computes reward function with added penalty for exceeding num needles fired 
      
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

        # Reward should sum to maximum of 1 * scale_factor (in this case 10 is chosen)
        if np.isnan(ccl_coeff):
          ccl_coeff = 0 
        
        print(f'CCL COEFF {ccl_coeff}')
        reward = (ccl_coeff * hit_rate * scale_factor) 
        #print(f'Reward {reward} ccl_coeff : {ccl_coeff} hit rate : {hit_rate} num_needles_fired : {num_needles_fired}')

      else:
        
        if needle_fired: 
            if needle_hit:
                reward = 100 
            else:
                reward = -1 
        
        # No needles fired, but still taking up time navigating 
        else:
            reward = -0.8
    
      # Penalty for firing more needles than max needles fired of -10 
      if max_num_needles_fired:
          reward -= penalty

      # Penalise for staying at the same position or moving off the grid
      if same_position: 
          reward -= 1 

      if moves_off_grid:
          reward -= 1

      return reward

    def compute_reward_hit_rate(self, done, ccl_coeff, hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid, miss_penalty = 2, penalty = 5, scale_factor = 100):
        
      """
      Computes reward function with added penalty for exceeding num needles fired 
      
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

        # Reward should sum to maximum of 1 * scale_factor (in this case 10 is chosen)
        if np.isnan(ccl_coeff):
          ccl_coeff = 0 
        
        #print(f'CCL COEFF {ccl_coeff}')
        reward = (hit_rate * scale_factor) 
        #print(f'Reward {reward} ccl_coeff : {ccl_coeff} hit rate : {hit_rate} num_needles_fired : {num_needles_fired}')

      else:

          if needle_hit:
              reward = 100 

          else:
              reward = -miss_penalty # -2 * 50 = -100 reward for not hitting any lesion at the end of 50 steps ie bad episode 
      
      # Penalty for firing more needles than max needles fired of -10 
      if max_num_needles_fired:
          reward -= penalty

      # Penalise for staying at the same position or moving off the grid
      if same_position: 
          reward -= 1 

      if moves_off_grid:
          reward -= 1

      return reward
 
    def compute_reward_ccl(self, done, ccl_coeff, hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid, miss_penalty = 2, penalty = 5, scale_factor = 100):
        
      """
      Bonus reward is CCL
      Computes reward function with added penalty for exceeding num needles fired 
      
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

        # Reward should sum to maximum of 1 * scale_factor (in this case 10 is chosen)
        if np.isnan(ccl_coeff):
          ccl_coeff = 0 
        
        #print(f'CCL COEFF {ccl_coeff}')
        reward = (ccl_coeff * scale_factor) 
        #print(f'Reward {reward} ccl_coeff : {ccl_coeff} hit rate : {hit_rate} num_needles_fired : {num_needles_fired}')

      else:

          if needle_hit:
              reward = 100 

          else:
              reward = -miss_penalty # -2 * 50 = -100 reward for not hitting any lesion at the end of 50 steps ie bad episode 
      
      # Penalty for firing more needles than max needles fired of -10 
      if max_num_needles_fired:
          reward -= penalty

      # Penalise for staying at the same position or moving off the grid
      if same_position: 
          reward -= 1 

      if moves_off_grid:
          reward -= 1

      return reward

    def compute_reward_ccl_only(self, done, ccl_coeff, hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid, inc_HR = False, inc_CCL = False ,scale_factor = 100):
      
      """Computes simple reward function 
      
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

      reward = 0 

      if done:

        if np.isnan(ccl_coeff):
          ccl_coeff = 0 
        #print(f'CCL COEFF {ccl_coeff}')

        # Penalty for exceeding number of needles fired 
        if max_num_needles_fired:
          reward -= 50
        
        if inc_HR: 
          #BONUS reward of 10 * hit_rate (max 10, minimum 0 )
          min_hit_threshold = 0.5 

          if hit_rate > min_hit_threshold: 
            reward += hit_rate * scale_factor
          else: 
            reward -= scale_factor # did not hit the minimum number of lesions 
          
        if inc_CCL: 
          reward += ccl_coeff * scale_factor 

      if same_position: 
          reward -= 5 

      if moves_off_grid:
          reward -= 5

      return reward

    def compute_reward_simple_reward(self, done, ccl_coeff, hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid, inc_HR = False, inc_CCL = False ,scale_factor = 100):
        
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

      reward = 0 

      if done:

        if np.isnan(ccl_coeff):
          ccl_coeff = 0 
        #print(f'CCL COEFF {ccl_coeff}')

        # Penalty for exceeding number of needles fired 
        if max_num_needles_fired:
          reward -= 50
        
        if inc_HR: 
          #BONUS reward of 10 * hit_rate (max 10, minimum 0 )
          min_hit_threshold = 0.5 

          if hit_rate > min_hit_threshold: 
            reward += hit_rate * scale_factor
          else: 
            reward -= scale_factor # did not hit the minimum number of lesions 
          
        reward += ccl_coeff * scale_factor 

      else:
        
        if needle_fired: 
            if needle_hit:
                reward = 10 
            else:
                reward = -1 
        
        # No needles fired, but still taking up time navigating 
        #else:
        #    reward = -0.8 

      # Penalise for staying at the same position or moving off the grid
      if same_position: 
          reward -= 5 

      if moves_off_grid:
          reward -= 5

      return reward

    def compute_reward_ccl_reward(self, done, ccl_coeff, hit_rate, needle_fired, needle_hit, max_num_needles_fired, same_position, moves_off_grid, scale_factor = 100):
        
      """
      Computes reward function with added penalty for exceeding num needles fired 
      
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

        # Reward should sum to maximum of 1 * scale_factor (in this case 10 is chosen)
        if np.isnan(ccl_coeff):
          ccl_coeff = 0 
        
        print(f'CCL COEFF {ccl_coeff}')
        reward = (ccl_coeff * hit_rate * scale_factor) 
        #print(f'Reward {reward} ccl_coeff : {ccl_coeff} hit rate : {hit_rate} num_needles_fired : {num_needles_fired}')

      else:
        
        if needle_fired: 
            if needle_hit:
                reward = 100 
            else:
                reward = -1 
        
        # No needles fired, but still taking up time navigating 
        else:
            reward = -0.5 
    
      # Penalty for firing more needles than max needles fired of -10 
      if max_num_needles_fired:
          reward -= 10 

      # Penalise for staying at the same position or moving off the grid
      if same_position: 
          reward -= 1 

      if moves_off_grid:
          reward -= 1

      return reward


    def compute_reward_BASIC(self, needle_fired, needle_hit, penalty_reward = 10):
        
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

      reward = 0 

      # Reward for firing : set to 100 (incentivise firing more)
      if needle_hit:
        reward = 100
      else: 
        # Penalty for misfiring : automatically set to 10 
        reward = -penalty_reward 

      return reward

    def compute_reward_BASIC_FIRED(self, needle_fired, needle_hit, done = False, penalty_reward = 10):
        
      """
    
      Computes reward function, simila to basic, but differentiates firing from non-firing!!!
      
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

      reward = 0 

      # Reward for firing : set to 100 (incentivise firing more)
      if needle_fired:
        if needle_hit:
          reward = 100
        else: 
          # Penalty for misfiring : automatically set to 10 
          reward = -penalty_reward # rather get # -10 if needle is fired where there's no lesion 
      
      else:
        if needle_hit:
          reward = -100 # changed from -20 to -100 to FORCE agent to fire if a lesion is observed 
        else:
          reward = 0 #changed from +10 to 0 ; 0 if needle is not fired where there's no lesion 

      # Check if done -> penalty for not firing 
      if done and self.num_needles <= 3: # 4 needles minimum per lesion
        reward -= 500 #massive penalty for not firing 

      return reward

    def compute_reward_BASIC_20(self, needle_fired, needle_hit, penalty_reward = 10):
        
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

      reward = 0 

      # Reward for firing : set to 100 (incentivise firing more)
      if needle_hit:
        reward = 50
      else: 
        # Penalty for misfiring : automatically set to 10 
        reward = -penalty_reward 

      

      return reward

    def compute_needle_traj(self, x_grid, y_grid, noise_added = False):  
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

        # Change x_grid, y_grid from lesion coords to image coords
        x_grid = (x_grid*2) + self.prostate_centroid[0] # x_grid and y_grid multiplied by 2 to account for 0.5x0.5x1 resolution of mri dataset
        y_grid = (y_grid*2) + self.prostate_centroid[1]
        
        # 16g/18g corresponds to 1.2mm, 1.6mm diameter ie 3 pixels taken up on x,y plane 
        needle_mask = np.zeros_like(self.img_data['mri_vol'])
        needle_mask[y_grid -1 : y_grid +2 , x_grid -1 : x_grid +2, 0:self.max_needle_depth] = 1

        #intersection_volume = needle_mask * self.img_data['tumour_mask']

        return needle_mask 

    def compute_ccl(self, needle_mask):
        """
        Computes CCL given needle mask and tumour masks 

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

if __name__ == '__main__':
        
    #Evaluating agent on training and testing data 
    ps_path = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality'
    csv_path = '/Users/ianijirahmae/Documents/PhD_project/rectum_pos.csv'

    #ps_path = '/raid/candi/Iani/MRes_project/Reinforcement Learning/DATASETS/'
    #rectum_path = '/raid/candi/Iani/MRes_project/Reinforcement Learning/rectum_pos.csv'
    
    log_dir = 'test'
    os.makedirs(log_dir, exist_ok=True)

    PS_dataset_train = Image_dataloader(ps_path, csv_path, use_all = False, mode  = 'train')
    Data_sampler_train = DataSampler(PS_dataset_train)

    Biopsy_env_init = TemplateGuidedBiopsy_penalty(Data_sampler_train, results_dir = log_dir, max_num_steps = 100, reward_fn = 'patient', obs_space = 'both') #Data_sampler_train,
    
    test_obs = Biopsy_env_init.reset()

    #from PIL import Image, ImageDraw, ImageFont
    #img = Image.fromarray(np.uint8(test_obs[:,:,0]*255))
    #test_str = "Needles left:" + str(100)
    #img_test = ImageDraw.Draw(img)    
    #img_test.text((3, 90), test_str, fill = (100))
    #img_array = np.array(img)
    #img_normalised = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    #plt.imshow(img_array)
    #img.show()


    #Biopsy_env_init = frame_stack_v1(Biopsy_env_init, 3)
    initial_obs = Biopsy_env_init.reset()

    for episode_num in range(5):

        Biopsy_env_init.reset()
        print(f"Episode num : {episode_num}")

        done_new_patient = False
        all_actions = [] 
        all_rewards = [] 
        time_counter = 0 

        while not done_new_patient:

            #action, _states = Agent.predict(obs, deterministic = True)

            #Random action
            action = Biopsy_env_init.action_space.sample()
            all_actions.append(action)

            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = Biopsy_env_init.step(action)
            done_new_patient = info['new_patient']
            all_rewards.append(reward)
            time_counter +=1 
        
        print(f"Max num needles : {info['max_num_needles']}")
        plt.figure()
        fig, axes = plt.subplots(1,4)
        axes[0].imshow(obs[0,:,:,0])
        axes[1].imshow(obs[:,:,25])
        axes[2].imshow(obs[:,:,50])
        axes[3].imshow(info['firing_grid'])
        print('Chicken')
        num_needles = np.sum(np.stack(all_actions)[:,2] >= 0)
        print(f'Number of needles fired : {num_needles} \n')

        #fig_plot = info['ccl_plots']
        #Biopsy_env_init.reset()

        #all_episode_ccl[episode_num] = info['ccl_corr']
        #print(f"Time_step {time} reward : {reward} ccl : {[info['all_ccl'][-1] if reward != -0.5 else 0]}") # action_z : {action[2]}")
        #print(f"Episode length {time_counter}, total reward : {np.sum(all_rewards)}, final_reward : {reward} ")
        #print(f"CCL corr: {info['ccl_corr_online']} Hit rate : {info['hit_rate']} Efficiency {info['efficiency']:03f} \n")
    
        #all_episode_len[episode_num] = time_counter
   
    print('Chicken')
    print('Chicken')

    plt.figure()
    plt.imshow(obs[:,:,50])
    

    seed = 1

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
                Biopsy_env_init = TemplateGuidedBiopsy_penalty(Data_sampler_train, env_num = rank_num, obs_space = 'images')
                #Biopsy_env = frame_stack_v1(Biopsy_env_init, 3)
                Biopsy_env = Biopsy_env_init
                Biopsy_env.reset()
                #Biopsy_env.seed(seed + rank)
                Biopsy_env.action_space.seed(seed + rank)
            
                # Wrap the env in a Monitor wrapper
                # to have additional training information
                monitor_path = os.path.join(monitor_dir, str(rank)) 

                # Create the monitor folder if needed
                if monitor_path is not None:
                    os.makedirs(monitor_dir, exist_ok=True)

                env = Monitor(Biopsy_env, filename=monitor_path)

                return env

            set_random_seed(seed)
            return _init        

        if vecenv_class == 'Dummy':
            return DummyVecEnv([make_env(i) for i in range(n_envs)])
        else: 
            return SubprocVecEnv([make_env(i) for i in range(n_envs)])
            
    Vec_Biopsy_env = make_vec_env(n_envs = 4, vecenv_class = 'Dummy', monitor_dir = 'test_cats')#.to(device_cuda)
    Biopsy_env = VecFrameStack(Vec_Biopsy_env, 3, channels_order = 'last')

    from stable_baselines3.common.vec_env import VecFrameStack
    Biopsy_env_init = VecFrameStack(Biopsy_env, 3)
    #for key,subspace in Biopsy_env.observation_space.spaces.items():
    #  print(key)

    policy_kwargs = dict(features_extractor_class = SimpleFeatureExtractor_3D, features_extractor_kwargs=dict(multiple_frames = True, num_multiple_frames = 75))
    agent = PPO(CnnPolicy, Biopsy_env, policy_kwargs = policy_kwargs, n_epochs = 2, learning_rate = 0.0001, tensorboard_log = 'test')
    callback_train = SaveOnBestTrainingRewardCallback_moreinfo(check_freq=100, log_dir = 'test')
    agent.learn(total_timesteps= 1000, callback = callback_train)

    Biopsy_env_init = frame_stack_v1(Biopsy_env_init, 3)
    initial_obs = Biopsy_env_init.reset()
    #initial_obs = Biopsy_env_init.reset()

    # Initialise random agent 
    policy_kwargs = dict(features_extractor_class = FeatureExtractor, features_extractor_kwargs=dict(multiple_frames = True, num_multiple_frames = 75))
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    Agent = PPO(CnnPolicy, Biopsy_env_init,  gamma = 0.9, policy_kwargs = policy_kwargs, \
        n_steps = 1000, batch_size = 128, n_epochs = 2, learning_rate = 0.0001, \
          ent_coef = 0.0001, tensorboard_log = log_dir)

     # Take steps in biopsy env to test new moving off grid 
    #obs, reward, done, info = Biopsy_env_init.step(np.array([-1,0,0]))

    # Initialise episode, let agent act randomly 
    all_episode_len = np.zeros(100,) 
    all_episode_ccl = np.zeros(100,)

