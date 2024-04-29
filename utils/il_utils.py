# from random import random

import torch
import numpy as np 
from torch.utils.data import Dataset
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os 
import copy 
import gym 
import pandas as pd 
import h5py 
#from utils_il import * 
from utils.rl_utils import * 
from torch.utils.tensorboard import SummaryWriter

class SimBiopsyEnv():
    """
    A simulated biopsy env, using functions from RL Biopsy env
    
    # Class methods: 
    find_new_needle_pos()
    
    obtain_new_obs()
    
    create_needle_vol()
    
    compute_HR()
    
    compute_CCL()
    
    """
    
    def __init__(self, device):
        self.device = device
        print(f"Device : {self.device}")
        
    def find_new_needle_pos(self, actions, grid_pos):
        """
        Obtains new grid positions, given current actions 
        
        Arguments:
        :actions (tensor): 3x1 tensor of chosen actions
        :prev_pos (tensor): 2x1 previous position 
        
        
        Returns:
        :new_pos (tensor) : new position on grid, along with suitable depth 
        """
        
        # Convert action z into depths 
        action_z = actions[:,2].to(self.device)
        depths = torch.zeros_like(action_z).to(self.device)
        non_fired = [action_z <=-0.33]
        apex = (action_z > -0.33) * (action_z <= 0.33)
        base = action_z > 0.33 
        
        depths[non_fired] = 0 
        depths[apex] = 1
        depths[base] = 2
        
        # Convert x,y actions 
        
        max_step_size = 10 # changed max step size to 10 
        grid_pos = grid_pos.to(self.device)
        prev_pos = grid_pos - torch.tensor([50,50]).to(self.device) # subtract by origin of 50,50 

        action_x = actions[:,0]
        action_y = actions[:,1]

        x_movement = round_to_05(action_x * max_step_size)
        y_movement = round_to_05(action_y * max_step_size)

        updated_x = prev_pos[:,0] + x_movement
        updated_y = prev_pos[:,1] + y_movement

        #Dealing with boundary positions 
        x_lower = updated_x < -30
        x_higher = updated_x > 30
        y_lower = updated_y < -30
        y_higher = updated_y > 30

        if torch.any(x_lower).item():
            #Change updated_x to maximum     
            #updated_x =  -30
            updated_x[x_lower] = -30

        if torch.any(x_higher).item():
            updated_x[x_higher] =  30

        if torch.any(y_lower).item(): 
            updated_y[y_lower] = -30

        if torch.any(y_higher).item(): 
            updated_y[y_higher] = 30

        x_grid = updated_x.int()
        y_grid = updated_y.int()

        new_pos = torch.stack([(x_grid), (y_grid), depths], dim = 1)

        return new_pos
            
    def obtain_new_obs(self, current_pos, max_depth):
      """
      Obtains new needle pos given the new grid position 

      Parameters:
      ----------
      current_pos : 3 x 1 array (x,y,z) where z is (0,1,2) where 0 i snon-fired, 1 is apex and 2 is base 
      
      Returns:
      ----------
      obs : batch_size x 100 x 100 x 25 observations 
      
      """
      batch_size = current_pos.size()[0]
      needle_obs = torch.zeros([batch_size, 100, 100, 24])
      for i in range(batch_size):
        grid_pos = current_pos[i,:]
        needle_obs[i, :,:,:] = self.create_needle_vol(grid_pos, max_depth[i])
    
      return needle_obs
   
    def create_needle_vol(self, current_pos, max_depth):

      """
      A function that creates needle volume 100 x 100 x 24 

      Parameters:
      -----------
      current_pos : current position on the grid. 1 x 3 array delta_x, delta_y, delta_z ; assumes in the range of (-30,30) ie actions are multipled by 5 already 

      Returns:
      -----------
      neeedle_vol : 100 x 100 x 24 needle vol for needle trajectory 

      """
      needle_vol = torch.zeros([100,100,24])

      x_idx = current_pos[0]#*5
      y_idx = current_pos[1]#*5

      #Converts range from (-30,30) to image grid array
      x_idx = (x_idx) + 50
      y_idx = (y_idx) + 50

      x_grid_pos = int(x_idx)
      y_grid_pos = int(y_idx)

      depth_map = {0 : 1, 1 : int(0.5*max_depth), 2 : max_depth}
      depth = depth_map[int(current_pos[2])]

      needle_vol[y_grid_pos-1:y_grid_pos+ 2, x_grid_pos-1:x_grid_pos+2, 0:depth] = 1

      return needle_vol 

    def compute_HR(self, lesion_obs, needle_obs, pred_actions):
        """
        Compute both HR and CCL using observations 

        Arguments:
        :lesion_obs (tensor) : batch_size x 100 x 100 x 24 
        :needle_obs (tensor) : batch_Size x 100 x 100 x 24
        :pred_actions : batch_size x3 

        """

        # clone actions and obs 
        obs_img = lesion_obs

        # current needle pos 
        batch_size = needle_obs.size()[0]
        lesion_obs = lesion_obs.to(self.device)
        needle_obs = needle_obs.to(self.device)
        # find which idx are fired
        fired_idx = torch.where(pred_actions[:,-1] >= -0.33)[0]
        NUM_FIRED = len(fired_idx)

        # no fired needles -> no hit arrays, no ccl arrays 
        if NUM_FIRED == 0:
            print(f'No fire needles : HR = 0, CCL = 0')
            ccl_array = torch.tensor([0], dtype = torch.float32)
            hit_array = torch.tensor([0], dtype = torch.float32) 
            
            return hit_array, ccl_array
        else:
            
            # compute intsersections between new needle obs and lesion 
            intersect = lesion_obs[fired_idx, :, :, :] * needle_obs[fired_idx,  :, :, :]
            all_lesions = lesion_obs[fired_idx, :,:,:]

            # Indices of all the hit patients 
            all_hit_idx = (torch.unique(torch.where(intersect)[0]))

            # Compute HR 
            HR = (len(all_hit_idx) / NUM_FIRED) * 100

            # hit ar
            hit_array = torch.zeros(len(fired_idx))
            hit_array[all_hit_idx] = 1
            ccl_array = [] 
            
            # Compute CCL:
            for hit_idx in range(NUM_FIRED):
                if hit_array[hit_idx] == 0: 
                    ccl = 0 
                    norm_ccl = 0 
                else: 
                    z_depths = torch.where(intersect[hit_idx, :,:,:])[-1]
                    min_z = torch.min(z_depths)
                    max_z = torch.max(z_depths)
                    
                    # lesion depths 
                    z_lesion = torch.where(all_lesions[hit_idx, :, :, :])[-1]
                    min_z_lesion = torch.min(z_lesion)
                    max_z_lesion = torch.max(z_lesion)
                
                    # added 1 : to account for each voxel being 0.25 in resolution; if only intersect in one voxel (8,8) then still ccl = 4mm  
                    ccl = (max_z+1 - min_z )*4
                    ccl_max = (max_z_lesion+1 - min_z_lesion )*4
                    
                    norm_ccl = ccl / ccl_max 

                ccl_array.append(norm_ccl)

                #print(f'{max_z} , {min_z}, {ccl}')

            CCL = torch.nanmean(torch.FloatTensor(ccl_array))
            ccl_std = torch.std(torch.FloatTensor(ccl_array))

            print(f"Hit rate : {HR} CCL {CCL} +/- {ccl_std} for num fired : {NUM_FIRED}")

            return torch.tensor(hit_array, dtype = torch.float32), torch.tensor(ccl_array, dtype = torch.float32)

        
def compute_mse_actions(pred, labels):
    """
    
    A function that computes the MSE for each action (x,y,z)
    
    Args:
        pred (tensor): batch_Size x 3 float
        labels (tensor): batch_Size x 3 float 
        
    returns x,y,z mse 
    """
    with torch.no_grad():
        squared_dif = (pred - labels)**2
        mse_actions = torch.mean(squared_dif, axis = 0)
        
    return mse_actions
            
def round_to_05(val):
    """
    A function that rounds to the nearest 5mm
    """
    #rounded_05 = round(val * 2) / 2
    rounded_05 =  (torch.round(val / 5)) * 5
    return rounded_05

def compute_new_obs(current_pos, actions, max_depth):
    """
    Compute new needle obs given actions 
    """

    grid_pos = current_pos.detach().clone().cpu()
    all_actions = actions.detach().clone().cpu() 
    new_pos = torch.zeros_like(grid_pos)
    new_pos[:, 0:2] = round_to_05((grid_pos[:,0:2] + all_actions[:,0:2]) * 2)#(because normalised between (-1,1) from (-2,2))

    # unnormalise action from (-1,1)to  (0,1,2)where 0 is non-fired, 1 is apex, 2 is base 
    all_0 = (all_actions[:,-1] <= -0.33) # 0 < x <= -0.33 : 0
    all_1 = ((all_actions[:,-1] > -0.33) * (all_actions[:,-1] <=0.33))*2    # 0.33 < x <= 0.33 : 1
    all_2 = (all_actions[:,-1] > 0.33)*3    # 0.33 <= x < 1 : 2
    all_depths = (all_0 + all_1 + all_2) - 1

    new_pos[:,-1] = all_depths

    # compute new needle obs 
    vol_creater = GridArray()

    # new needle obs
    BATCH_SIZE = np.shape(new_pos)[0]

    all_vols = np.zeros((BATCH_SIZE, 100,100,24))
    for i in range(BATCH_SIZE):
        needle_vol = vol_creater.create_needle_vol(new_pos[i], max_depth[i])
        all_vols[i, :, :, :] = needle_vol

    return all_vols

def compute_grid_pos(obs):
    """
    Comptues curernt grid position, based on needle obs 
    To be used for comptuing HR / CCL after!!! 
    Args:
        obs (_type_): _description_
    """
    
    needle_mask = obs[:,-1,:,:,:]
    batch_size = needle_mask.size()[0]
    
    current_grid_pos = torch.zeros([batch_size, 2])
    for i in range(batch_size):
        y,x,_ = torch.where(needle_mask[i, :,:])
        current_grid_pos[i, :] = torch.tensor([torch.mean(x.to(torch.float)), torch.mean(y.to(torch.float))])
    
    return current_grid_pos 
    
class GridArray():
    
    def create_grid_array(self, action,  grid_array = None, display_current_pos = True):

        """
        
        A function that generates grid array coords

        Note: assumes that x_idx, y_idx are in the range (-30,30)

        """
        x_idx = action[1]*5
        y_idx = action[0]*5
        needle_fired = (action[2] == 1)

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

        return grid_array, self.saved_grid

    def create_needle_vol(self, action, max_depth):
        """
        A function that creates needle volume 100 x 100 x 24 
        """
        
        needle_vol = np.zeros([100,100,24])

        x_idx = action[1]*5
        y_idx = action[0]*5
        needle_fired = (action[2] == 1)

        #Converts range from (-30,30) to image grid array
        x_idx = (x_idx) + 50
        y_idx = (y_idx) + 50

        x_grid_pos = int(x_idx)
        y_grid_pos = int(y_idx)

        depth_map = {0 : 1, 1 : int(0.5*max_depth), 2 : max_depth}
        depth = depth_map[int(action[2])]

        needle_vol[y_grid_pos-1:y_grid_pos+ 2, x_grid_pos-1:x_grid_pos+2, 0:depth ] = 1

        return needle_vol 

    def create_needle_vol_wacky(self, action, max_depth):
        
        """
        A function that creates needle volume 100 x 100 x 24 
        """
        
        needle_vol = np.zeros([100,100,24])

        x_idx = action[0]*5
        y_idx = action[1]*5
        needle_fired = (action[2] == 1)

        #Converts range from (-30,30) to image grid array
        x_idx = (x_idx) + 50
        y_idx = (y_idx) + 50

        x_grid_pos = int(x_idx)
        y_grid_pos = int(y_idx)

        depth_map = {0 : 1, 1 : int(0.5*max_depth), 2 : max_depth}
        depth = depth_map[int(action[2])]

        needle_vol[y_grid_pos-1:y_grid_pos+ 2, x_grid_pos-1:x_grid_pos+2, 0:depth ] = 1

        return needle_vol 

def find_unique_values(lst):
    """
    Find the unique values in a list while preserving their order of appearance.

    Parameters:
    lst (list or array-like): The input list or array.

    Returns:
    list: A list containing the unique values in the order of their appearance.

    Example:
    >>> my_list = [2, 4, 2, 1, 3, 4, 1, 5]
    >>> unique_values = find_unique_values(my_list)
    >>> print(unique_values)
    [2, 4, 1, 3, 5]
    """

    unique_values, indices = np.unique(lst, return_index=True)
    unique_values_ordered = unique_values[np.argsort(indices)]
    return unique_values_ordered.tolist()

def compute_DIST(obs, actions, tumour_centroid):
    """
    Computes distance to closest lesion and whether agents' step is closer or further away than current position 
    TODO: add index of lesion targeted by action to action labels!!! to use for compute_dist_to_lesion metric 
    
    """
    
    raw_actions = torch.clamp(actions, -1,1).detach().clone().cpu().numpy()
    obs_img = obs.detach().clone().cpu().numpy()

    needle_mask = obs_img[:,-1,:,:,0]
    batch_size = np.shape(needle_mask)[0]

    current_grid_pos = np.zeros([batch_size, 2])
    for i in range(batch_size):
        y,x = np.where(needle_mask[i, :,:])
        current_grid_pos[i, :] = [np.mean(y), np.mean(x)]

    scaled_actions = (raw_actions[:, :-1] * 5 * 2) # only using x, y; multiply by 5 to get 5mm intervals; multiply by 2 because previously normalised by dividing by 2 so unormalising now 
    new_grid_pos = (scaled_actions + current_grid_pos) # y x z 

    # Compute distance between new position (current + action) and target lesion 
    # change tumour_centroid from x y z to y x z
    swapped_centres = tumour_centroid[:, [1,0,2]]

    # compute distance for each row 
    dist_t1 = np.linalg.norm(swapped_centres[:,0:2] - new_grid_pos, axis = 1) 
    dist_t0 = np.linalg.norm(swapped_centres[:,0:2] - current_grid_pos, axis = 1) 

    # sign : (dist_t+1 - dist_t) if +ve -> moving closer. If not, moving further away! --> gives indication of gradient. 
    sign_dist = np.sign(dist_t0 - dist_t1) # if t0 > t1 -> getting closer to centroid 
    #print(f'Sign(dist_t - dist_t+1 : {sign_dist} average : {np.mean(sign_dist)} +- {np.std(sign_dist)}')

    return sign_dist, np.mean(sign_dist)

def compute_HR(obs, actions, grid_pos, max_depth):
    """
    Compute both HR and CCL using observations 

    obs : (batch_size x 5 x 100 x 100 x 24)

    """

    # clone actions and obs 
    obs_img = obs.detach().clone().cpu()

    # clip actions between (-1,1)
    #pred_actions = (torch.clamp(actions, -1, 1).cpu())
    pred_actions = actions.detach().clone().cpu() 

    lesion = obs_img[:,0,:,:,:] 
    needle = obs_img[:,4,:,:,:]

    # current needle pos 
    needle_mask = obs_img[:,4,:,:,:]
    batch_size = np.shape(needle_mask)[0]

    # # Compute current grid pos from observation 
    # current_grid_pos = np.zeros([batch_size, 2])
    # new_grid_pos = np.zeros([batch_size, 2])

    # for i in range(batch_size):
    #     y,x,_ = np.where(needle_mask[i, :,:])
    #     current_grid_pos[i, :] = [np.mean(y), np.mean(x)]   

    #TODO : COMPUTE NEW OBS
    new_needle_obs = compute_new_obs(grid_pos, actions, max_depth)

    # find which idx are fired
    fired_idx = np.where(pred_actions[:,-1] >= -0.33)[0]
    NUM_FIRED = len(fired_idx)

    # no fired needles -> no hit arrays, no ccl arrays 
    if NUM_FIRED == 0:
        print(f'No fire needles : HR = 0, CCL = 0')
        ccl_array = [0]
        hit_array = [0]
        return hit_array, ccl_array
    else:
        
        # compute intsersections between new needle obs and lesion 
        intersect = lesion[fired_idx, :, :, :] * new_needle_obs[fired_idx,  :, :, :]

        # Indices of all the hit patients 
        all_hit_idx = (np.unique(np.where(intersect)[0]))

        # Compute HR 
        HR = (len(all_hit_idx) / NUM_FIRED) * 100

        # hit ar
        hit_array = np.zeros(len(fired_idx))
        hit_array[all_hit_idx] = 1

        ccl_array = [] 
        # Compute CCL:
        
        for hit_idx in range(NUM_FIRED):
            if hit_array[hit_idx] == 0: 
                ccl = 0 
            else: 
                z_depths = np.where(intersect[hit_idx, :,:,:])[-1]
                min_z = np.min(z_depths)
                max_z = np.max(z_depths)

                # added 1 : to account for each voxel being 0.25 in resolution; if only intersect in one voxel (8,8) then still ccl = 4mm  
                ccl = (max_z+1 - min_z )*4

            ccl_array.append(ccl)

            #print(f'{max_z} , {min_z}, {ccl}')

        CCL = np.mean(ccl_array)
        ccl_std = np.std(ccl_array)
        
        print(f"Hit rate : {HR} CCL {CCL} +/- {ccl_std}")

        return hit_array, ccl_array

def compute_metrics_gs(obs, actions, grid_pos, max_depth):
    """
    Computes HR and CCL from new action and observations 
    """
class TimeStep_data(Dataset):

    def __init__(self, folder_name, labels_path = 'action_labels.h5', mode = 'train', finetune = False):

        self.folder_name = folder_name
        self.mode = mode
        self.labels_path = labels_path
        self.finetune = finetune

        # Obtain list of patient names with multiple lesions -> change to path name
        df_dataset = pd.read_csv('/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv')
        #df_dataset = pd.read_csv('/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv')
        #Filter out patients >=5 lesions 
        patients_w5 = np.where(df_dataset[' num_lesions'] >= 5)[0] # save these indices for next time!!!
        # Remove patients where lesions >5 as these are incorrectly labelled!!
        df_dataset = df_dataset.drop(df_dataset.index[patients_w5])
    

        self.all_file_names = df_dataset['patient_name'].tolist()
        self.num_lesions = df_dataset[' num_lesions'].tolist()

        # read h5 file 
        self.grid_labels = h5py.File(labels_path, 'r')
        size_dataset = len(self.all_file_names)
        train_len = int(size_dataset * 0.7) 
        test_len = int(size_dataset * 0.2) 
        val_len = size_dataset - (train_len + test_len)

        # both test and val have simila rnumber of lesions (mean = 2.4 lesions)
        self.train_names = self.all_file_names[0:train_len]
        self.val_names = self.all_file_names[train_len:train_len + val_len]
        self.test_names = self.all_file_names[train_len + val_len:]
        self.dataset_len = {'train' : train_len, 'test': test_len, 'val' : val_len}

        # Folder names
        self.lesion_folder = os.path.join(folder_name, 'lesion')
        self.mri_folder = os.path.join(folder_name, 't2w')
        self.prostate_folder = os.path.join(folder_name, 'prostate_mask')

    def _normalise(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = np.max(img)
        min_img = np.min(img)

        #Normalise values between 0 to 1
        normalised_img =  ((img - min_img)/(max_img - min_img)) 
        return normalised_img.astype(np.float32)
    
    def _normalise_actions(self, actions):

        normalised_actions = np.zeros_like(actions)

        # First two actions x,y in delta range (-2,2)
        normalised_actions[0:2] = actions[0:2] / 2
        normalised_actions[-1] = (actions[-1] - 0.5) *2 

        return normalised_actions 

    def __len__(self):
        return self.dataset_len[self.mode]
 
    def __getitem__(self, idx):
        """
        Gets items from train, test, val 

        # From each patient idx 
            # randomly sample 3 sequential images starting from i = 1 
            # randomly sample 3 images 

        Returns:
        ------------
        template_grid : 100 x 100 x 25 (dataset)
        actions: 
        """

        if self.mode == 'train':
            #idx_ = idx
            patient_name = self.train_names[idx]

        elif self.mode == 'val':
            #idx_ = idx + self.dataset_len['train']
            patient_name = self.val_names[idx]

        elif self.mode == 'test':
            #idx_ = idx + self.dataset_len['train'] + self.dataset_len['val']
            patient_name = self.test_names[idx]


        # Read prostate mask, lesion mask, prostate mask separately using ImageReader    
        #patient_name = self.all_file_names[idx_]
        read_img = ImageReader()    
        mri_vol = np.transpose(self._normalise(read_img(os.path.join(self.mri_folder, patient_name))), [1, 2, 0])
        lesion_mask = np.transpose(self._normalise(read_img(os.path.join(self.lesion_folder, patient_name))), [1, 2, 0])
        prostate_mask = np.transpose(self._normalise(read_img(os.path.join(self.prostate_folder, patient_name))), [1, 2, 0])
        
        # Obtain combined prostate_lesion and turn into torch tensor
        combined_mask = (torch.from_numpy(self.get_img_mask(prostate_mask, lesion_mask))[0::2, 0::2, 0::4])/2
        
        # Get file path name 
        sitk_img_path = os.path.join(self.lesion_folder, patient_name)

        # Read grid image label 
        patient_file = self.grid_labels[patient_name]
        all_actions = np.array(patient_file['all_actions'])
        all_grids = np.array(patient_file['all_grids'])
        NUM_ACTIONS = np.shape(all_actions)[0]
        
        # Randomly sample idx from 0: T-3 (num_actions-2 because anrage aranges from 0 to n-1 val)
        if self.finetune: 
            random_idx = 0 # finetune only with first frames, to let agent know how to start the actions with different datasets!!! 
        else:
            random_idx = np.random.choice(np.arange(0, NUM_ACTIONS-1))

        #print(f'Len of actions : {NUM_ACTIONS} random idx : {random_idx}')
        #sampled_actions = all_actions[random_idx:random_idx+3, :]
        final_action = all_actions[random_idx,:] # Only consider final action to be estimated
        
        # Fixed from time step T - 2 : T instead of T : T+2
        if random_idx == 0:
            sampled_grid = np.array([np.zeros((100,100)), np.zeros((100,100)), np.array(all_grids[random_idx])])
        elif random_idx == 1:
            sampled_grid = np.array([np.zeros((100,100)), np.array(all_grids[random_idx-1]), np.array(all_grids[random_idx])])
        else:
            sampled_grid = all_grids[random_idx - 2 :random_idx+1 , :]
            
        combined_grid = np.zeros((100,100,75))
        
        for i in range(3): 
            idx = i*25
            combined_grid[:,:,idx] = sampled_grid[i,:,:]
            combined_grid[:,:,idx+1: (i+1)*25] = combined_mask 
            #print(f'Idx : {idx}')
            #print(f' Idx : {idx+1} : {(i+1)*25}')

        # Combined grid : Contains template grid pos chosen at time steps t-3 : T
        # Final action : Action to take from time step T 
        combined_grid = torch.unsqueeze(torch.tensor(combined_grid), axis = 0)
        final_action = torch.tensor(self._normalise_actions(final_action))#, axis = 0)

        # Combined actions 
        
        return combined_grid, final_action 
    
    def get_img_mask(self, prostate_mask, tumour_mask):
        """
        Adds up the prostate and tumour mask together to obtain binary masks of these two objects together
        """

        prostate_vol = prostate_mask[:, :, :] #prostate = 1
        tumour_vol = tumour_mask[:, :, :] * 2 #tumour = 2
        combined_tumour_prostate = prostate_vol + tumour_vol

        # if lesion and prostate intersection, keep us lesion label 
        combined_tumour_prostate[combined_tumour_prostate >= 2] = 2

        return combined_tumour_prostate 

class TimeStep_data_moreinfo(Dataset):

    def __init__(self, folder_name, csv_path, labels_path = 'action_labels.h5', mode = 'train', finetune = False, single_patient = False):

        self.folder_name = folder_name
        self.mode = mode
        self.labels_path = labels_path
        self.finetune = finetune
        self.single_patient = single_patient

        # Obtain list of patient names with multiple lesions -> change to path name
        #df_dataset = pd.read_csv('/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv')
        #df_dataset = pd.read_csv('/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv')
        df_dataset = pd.read_csv(csv_path)
        self.all_file_names = df_dataset['patient_name'].tolist()
        self.num_lesions = df_dataset[' num_lesions'].tolist()

        # read h5 file 
        self.grid_labels = h5py.File(labels_path, 'r')
        size_dataset = len(self.all_file_names)
        train_len = int(size_dataset * 0.7) 
        test_len = int(size_dataset * 0.2) 
        val_len = size_dataset - (train_len + test_len)

        # both test and val have simila rnumber of lesions (mean = 2.4 lesions)
        self.train_names = self.all_file_names[0:train_len]
        self.val_names = self.all_file_names[train_len:train_len + val_len]
        self.test_names = self.all_file_names[train_len + val_len:]
        self.dataset_len = {'train' : train_len, 'test': test_len, 'val' : val_len}

        # Folder names
        self.lesion_folder = os.path.join(folder_name, 'lesion')
        self.mri_folder = os.path.join(folder_name, 't2w')
        self.prostate_folder = os.path.join(folder_name, 'prostate_mask')

    def _normalise(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = np.max(img)
        min_img = np.min(img)

        #Normalise values between 0 to 1
        normalised_img =  ((img - min_img)/(max_img - min_img)) 

        return normalised_img.astype(np.float32)
    
    def _normalise_actions(self, actions):

        normalised_actions = np.zeros_like(actions)

        # First two actions x,y in delta range (-2,2)
        normalised_actions[0:2] = actions[0:2] / 2
        normalised_actions[-1] = (actions[-1] - 0.5) *2 

        return normalised_actions 

    def __len__(self):
        return self.dataset_len[self.mode]
 
    def __getitem__(self, idx):
        """
        Gets items from train, test, val 

        # From each patient idx 
            # randomly sample 3 sequential images starting from i = 1 
            # randomly sample 3 images 

        Returns:
        ------------
        template_grid : 100 x 100 x 25 (dataset)
        actions: 
        """

        if self.mode == 'train':
            #idx_ = idx
            if self.single_patient: 
                idx = 1
            patient_name = self.train_names[idx]

        elif self.mode == 'val':
            #idx_ = idx + self.dataset_len['train']
            patient_name = self.val_names[idx]

        elif self.mode == 'test':
            #idx_ = idx + self.dataset_len['train'] + self.dataset_len['val']
            patient_name = self.test_names[idx]


        # Read prostate mask, lesion mask, prostate mask separately using ImageReader    
        #patient_name = self.all_file_names[idx_]
        read_img = ImageReader()    
        mri_vol = np.transpose(self._normalise(read_img(os.path.join(self.mri_folder, patient_name))), [1, 2, 0])
        lesion_mask = np.transpose(self._normalise(read_img(os.path.join(self.lesion_folder, patient_name))), [1, 2, 0])
        prostate_mask = np.transpose(self._normalise(read_img(os.path.join(self.prostate_folder, patient_name))), [1, 2, 0])
        
        # Obtain combined prostate_lesion and turn into torch tensor
        combined_mask = (torch.from_numpy(self.get_img_mask(prostate_mask, lesion_mask))[0::2, 0::2, 0::4])/2
        
        # Get file path name 
        sitk_img_path = os.path.join(self.lesion_folder, patient_name)

        # Read grid image label 
        patient_file = self.grid_labels[patient_name]
        all_actions = np.array(patient_file['all_actions'])
        all_grids = np.array(patient_file['all_grids'])
        NUM_ACTIONS = np.shape(all_actions)[0]
        
        # Randomly sample idx from 0: T-3 (num_actions-2 because anrage aranges from 0 to n-1 val)
        if self.finetune: 
            random_idx = 0 # finetune only with first frames, to let agent know how to start the actions with different datasets!!! 
        else:
            random_idx = np.random.choice(np.arange(0, NUM_ACTIONS-1))

        #print(f'Len of actions : {NUM_ACTIONS} random idx : {random_idx}')
        #sampled_actions = all_actions[random_idx:random_idx+3, :]
        final_action = all_actions[random_idx,:] # Only consider final action to be estimated
        
        # Fixed from time step T - 2 : T instead of T : T+2
        if random_idx == 0:
            sampled_grid = np.array([np.zeros((100,100)), np.zeros((100,100)), np.array(all_grids[random_idx])])
        elif random_idx == 1:
            sampled_grid = np.array([np.zeros((100,100)), np.array(all_grids[random_idx-1]), np.array(all_grids[random_idx])])
        else:
            sampled_grid = all_grids[random_idx - 2 :random_idx+1 , :]
            
        combined_grid = np.zeros((100,100,75))
        
        for i in range(3): 
            idx = i*25
            combined_grid[:,:,idx] = sampled_grid[i,:,:]
            combined_grid[:,:,idx+1: (i+1)*25] = combined_mask 
            #print(f'Idx : {idx}')
            #print(f' Idx : {idx+1} : {(i+1)*25}')

        # Combined grid : Contains template grid pos chosen at time steps t-3 : T
        # Final action : Action to take from time step T 
        combined_grid = torch.unsqueeze(torch.tensor(combined_grid), axis = 0)
        final_action = torch.tensor(self._normalise_actions(final_action))#, axis = 0)

        # Additional informaiton for metrics computation 
        downsampled_lesion_mask = lesion_mask[::2, ::2, ::4]
        lesion_centroids = np.array(patient_file['lesion_centroids']) # lesion centroids for computing distance metric
        all_identifiers = np.array(patient_file['action_identifiers']) # index of lesions to visit 
        action_identifier = int(all_identifiers[0,random_idx])
        tumour_centroid = lesion_centroids[action_identifier,:]

        # Combined actions 
        
        return combined_grid, final_action, downsampled_lesion_mask, action_identifier, tumour_centroid
    
    def get_img_mask(self, prostate_mask, tumour_mask):
        """
        Adds up the prostate and tumour mask together to obtain binary masks of these two objects together
        """

        prostate_vol = prostate_mask[:, :, :] #prostate = 1
        tumour_vol = tumour_mask[:, :, :] * 2 #tumour = 2
        combined_tumour_prostate = prostate_vol + tumour_vol

        # if lesion and prostate intersection, keep us lesion label 
        combined_tumour_prostate[combined_tumour_prostate >= 2] = 2

        return combined_tumour_prostate 

class TimeStep_data_new(Dataset):

    def __init__(self, folder_name, csv_path= 'csv_file.csv', labels_path = 'action_labels.h5', mode = 'train', finetune = False, single_patient = False):

        self.folder_name = folder_name
        self.mode = mode
        self.labels_path = labels_path
        self.finetune = finetune
        self.single_patient = single_patient
        
        # Obtain list of patient names with multiple lesions -> change to path name
        #df_dataset = pd.read_csv('/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv')
        #df_dataset = pd.read_csv('/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv')
        df_dataset = pd.read_csv(csv_path)
        self.all_file_names = df_dataset['patient_name'].tolist()
        self.num_lesions = df_dataset[' num_lesions'].tolist()

        self.vol_creater = GridArray() 

        # read h5 file 
        self.grid_labels = h5py.File(labels_path, 'r')
        size_dataset = len(self.all_file_names)
        train_len = int(size_dataset * 0.7) 
        test_len = int(size_dataset * 0.2) 
        val_len = size_dataset - (train_len + test_len)

        # both test and val have simila rnumber of lesions (mean = 2.4 lesions)
        self.train_names = self.all_file_names[0:train_len]
        self.val_names = self.all_file_names[train_len:train_len + val_len]
        self.test_names = self.all_file_names[train_len + val_len:]
        self.dataset_len = {'train' : train_len, 'test': test_len, 'val' : val_len}

        # Folder names
        self.lesion_folder = os.path.join(folder_name, 'lesion')
        self.mri_folder = os.path.join(folder_name, 't2w')
        self.prostate_folder = os.path.join(folder_name, 'prostate_mask')

    def _normalise(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = np.max(img)
        min_img = np.min(img)

        #Normalise values between 0 to 1
        normalised_img =  ((img - min_img)/(max_img - min_img)) 

        return normalised_img.astype(np.float32)
    
    def _normalise_actions(self, actions):

        normalised_actions = np.zeros_like(actions)

        # First two actions x,y in delta range (-2,2)
        normalised_actions[0:2] = actions[0:2] / 2
        #normalised_actions[-1] = (actions[-1] - 0.5) *2 
        normalised_actions[-1] = actions[2] - 1

        return normalised_actions 

    def __len__(self):
        return self.dataset_len[self.mode]
 
    def __getitem__(self, idx):
        """
        Gets items from train, test, val 

        # From each patient idx 
            # randomly sample 3 sequential images starting from i = 1 
            # randomly sample 3 images 

        Returns:
        ------------
        template_grid : 100 x 100 x 25 (dataset)
        actions: 
        """

        if self.mode == 'train':
            #idx_ = idx
            if self.single_patient: 
                idx = 1
            patient_name = self.train_names[idx]

        elif self.mode == 'val':
            #idx_ = idx + self.dataset_len['train']
            patient_name = self.val_names[idx]

        elif self.mode == 'test':
            #idx_ = idx + self.dataset_len['train'] + self.dataset_len['val']
            patient_name = self.test_names[idx]


        # Read prostate mask, lesion mask, prostate mask separately using ImageReader    
        #patient_name = self.all_file_names[idx_]
        read_img = ImageReader()    
        mri_vol = np.transpose(self._normalise(read_img(os.path.join(self.mri_folder, patient_name))), [1, 2, 0])
        lesion_mask = np.transpose(self._normalise(read_img(os.path.join(self.lesion_folder, patient_name))), [1, 2, 0])
        prostate_mask = np.transpose(self._normalise(read_img(os.path.join(self.prostate_folder, patient_name))), [1, 2, 0])
        
        # downsampled prostate mask 
        ds_prostate_mask = prostate_mask[::2,::2,::4]
        max_depth = np.max(np.where(ds_prostate_mask == 1)[-1])
        ds_lesion_mask = lesion_mask[::2,::2,::4]

        # Obtain combined prostate_lesion and turn into torch tensor
        combined_mask = (torch.from_numpy(self.get_img_mask(prostate_mask, lesion_mask))[0::2, 0::2, 0::4])/2
        
        # Get file path name 
        sitk_img_path = os.path.join(self.lesion_folder, patient_name)

        # Read grid image label 
        patient_file = self.grid_labels[patient_name]
        all_actions = np.array(patient_file['all_actions'])
        all_grids = np.array(patient_file['all_grids'])
        all_pos = np.array(patient_file['current_pos'])
        NUM_ACTIONS = np.shape(all_actions)[0]

        # Create needle vol array for all positions 

        needle_vol = []
        for pos_idx in range(NUM_ACTIONS):
            current_pos = all_pos[pos_idx,:]
            needle_vol.append(self.vol_creater.create_needle_vol(current_pos, max_depth))

        # Randomly sample idx from 0: T-3 (num_actions-2 because anrage aranges from 0 to n-1 val)
        if self.finetune: 
            random_idx = 0 # finetune only with first frames, to let agent know how to start the actions with different datasets!!! 
        else:
            random_idx = np.random.choice(np.arange(0, NUM_ACTIONS-1))

        #print(f'Len of actions : {NUM_ACTIONS} random idx : {random_idx}')
        #sampled_actions = all_actions[random_idx:random_idx+3, :]
        final_action = all_actions[random_idx,:] # Only consider final action to be estimated
        
        # Fixed from time step T - 2 : T instead of T : T+2
        if random_idx == 0:
            #sampled_grid = np.array([np.zeros((100,100)), np.zeros((100,100)), np.array(all_grids[random_idx])])
            needle_stack = torch.tensor(np.array([np.zeros([100,100,24]), np.zeros([100,100,24]), needle_vol[random_idx]]))
        elif random_idx == 1:
            #sampled_grid = np.array([np.zeros((100,100)), np.array(all_grids[random_idx-1]), np.array(all_grids[random_idx])])
            needle_stack = torch.tensor(np.array([np.zeros([100,100,24]), needle_vol[random_idx-1], needle_vol[random_idx]]))
        else:
            #sampled_grid = all_grids[random_idx - 2 :random_idx+1 , :]
            needle_stack = torch.tensor(np.array(needle_vol[random_idx-2 : random_idx+1]))
            
        # Combined grid : Contains template grid pos chosen at time steps t-3 : T
        # Final action : Action to take from time step T 
        final_action = torch.tensor(self._normalise_actions(final_action))#, axis = 0)

        # Additional informaiton for metrics computation 
        downsampled_lesion_mask = lesion_mask[::2, ::2, ::4]
        lesion_centroids = np.array(patient_file['lesion_centroids']) # lesion centroids for computing distance metric
        all_identifiers = np.array(patient_file['action_identifiers']) # index of lesions to visit 
        action_identifier = int(all_identifiers[0,random_idx])
        tumour_centroid = lesion_centroids[action_identifier,:]

        ## Obtain observation of needle volumes: 
        obs_stack = torch.cat((torch.tensor(ds_lesion_mask).unsqueeze(0), torch.tensor(ds_prostate_mask).unsqueeze(0), needle_stack), axis = 0)

        #obs_stack = torch.tensor(needle_vol[random_idx-2 : random_idx+1])

        # Combined actions 

        # Obtain actions and current position at time step chosen 
        grid_pos = all_pos[random_idx, :]
        
        return obs_stack, final_action, downsampled_lesion_mask, action_identifier, tumour_centroid, grid_pos, max_depth 
    
    def get_img_mask(self, prostate_mask, tumour_mask):
        """
        Adds up the prostate and tumour mask together to obtain binary masks of these two objects together
        """

        prostate_vol = prostate_mask[:, :, :] #prostate = 1
        tumour_vol = tumour_mask[:, :, :] * 2 #tumour = 2
        combined_tumour_prostate = prostate_vol + tumour_vol

        # if lesion and prostate intersection, keep us lesion label 
        combined_tumour_prostate[combined_tumour_prostate >= 2] = 2

        return combined_tumour_prostate 

class Timestep_data_GS(Dataset):
    
    def __init__(self, folder_name, csv_path= 'csv_file.csv', labels_path = 'action_labels.h5', mode = 'train', single_patient = False, num_patients = 'all'):
        
        self.folder_name = folder_name
        self.mode = mode
        self.labels_path = labels_path
        self.single_patient = single_patient
        self.num_patients = num_patients

        # Obtain list of patient names with multiple lesions -> change to path name
        #df_dataset = pd.read_csv('/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv')
        #z3df_dataset = pd.read_csv('/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv')
        df_dataset = pd.read_csv(csv_path)
        
        #Filter out patients >=5 lesions 
        patients_w5 = np.where(df_dataset[' num_lesions'] >= 5)[0] # save these indices for next time!!!
        df_dataset = df_dataset.drop(df_dataset.index[patients_w5])
        self.all_file_names = df_dataset['patient_name'].tolist()
        self.num_lesions = df_dataset[' num_lesions'].tolist()
  
        # read h5 file 
        self.grid_labels = h5py.File(labels_path, 'r')
        size_dataset = len(self.all_file_names)
        train_len = int(size_dataset * 0.7) 
        # test_len = int(size_dataset * 0.2)
        test_len = 0
        val_len = size_dataset - (train_len + test_len)

        # both test and val have simila rnumber of lesions (mean = 2.4 lesions)
        self.train_names = self.all_file_names[0:train_len]
        self.val_names = self.all_file_names[train_len:train_len + val_len]
        self.test_names = self.all_file_names[train_len + val_len:]
        self.dataset_len = {'train' : train_len, 'test': test_len, 'val' : val_len}
        self.prostate_folder = os.path.join(folder_name, 'prostate_mask')
        
    def __getitem__(self, idx):
        """
        Gets items from train, test, val 

        Returns:
        ------------
        template_grid : 100 x 100 x 25 (dataset)
        actions: 
        """

        if self.mode == 'train':
            #idx_ = idx
            #print(f'train')
            if self.single_patient: 
                idx = 1
            
            use_all_patients = (self.num_patients == 'all')
            
            if not use_all_patients: 
                #print(f"Num patients used for training: {self.num_patients}")
                
                if self.num_patients == '1':
                    idx = 1
                else: 
                    idx = np.random.choice(range(0,int(self.num_patients)))
                    #print(f"idx : {idx}")

            patient_name = self.train_names[idx]

        elif self.mode == 'val':
            #idx_ = idx + self.dataset_len['train']
            #print(f'val')
            patient_name = self.val_names[idx]
            #print(f'val {patient_name}')

        elif self.mode == 'test':
            #idx_ = idx + self.dataset_len['train'] + self.dataset_len['val']
            patient_name = self.test_names[idx]
            #print(f'test {patient_name}')

        # TODO: 
        # 1. FROM H5PY FILE, GET OBSERVAITON, ACTIONS
        # if patient_name == 'Patient418081829_study_0.nii.gz':
        #     print(f"Patient418081829_study_0has idx : {idx} ")
        #     # Sample new idx not 0
        #     new_idx = np.random.choice((np.delete(np.arange(0,self.dataset_len[self.mode]), idx)))
        #     print(f"Sampling new random idx {new_idx}")
        #
        #     if self.mode == 'test':
        #         patient_name = self.test_names[new_idx]
        #     if self.mode == 'val':
        #         patient_name = self.val_names[new_idx]
        #     if self.mode == 'train':
        #         patient_name = self.train_names[new_idx]
        #
        while not(patient_name in self.grid_labels):

            print(f'Patient name {patient_name} with idx :{idx} not in dataset, using random index instead')
            new_idx = np.random.choice((np.delete(np.arange(0,self.dataset_len[self.mode]), idx)))
            print(f"Random idx {new_idx}")

            if self.mode == 'test':
                patient_name = self.test_names[new_idx]
            if self.mode == 'val':
                patient_name = self.val_names[new_idx]
            if self.mode == 'train':
                patient_name = self.train_names[new_idx]
                
                
        patient_info = self.grid_labels[patient_name]
        # else: 
        #     print(f'Patient name {patient_name} with idx :{idx} not in dataset, using random index instead')
        #     new_idx = np.random.choice((np.delete(np.arange(0,self.dataset_len[self.mode]), idx)))
        #     print(f"Random idx {new_idx}")
            
        #     if self.mode == 'test':
        #         patient_name = self.test_names[new_idx]
        #     if self.mode == 'val':
        #         patient_name = self.val_names[new_idx]
        #     if self.mode == 'train':
        #         patient_name = self.train_names[new_idx]
                
        #     patient_info = self.grid_labels[patient_name]
            
        # Check if patient in list of patient names: 
        all_patient_names = os.listdir(self.prostate_folder)
        if not(patient_name in all_patient_names):
            new_idx = np.random.choice((np.delete(np.arange(0,self.dataset_len[self.mode]), idx)))
            print(f"Sampling new random idx {new_idx}")
            
            if self.mode == 'test':
                patient_name = self.test_names[new_idx]
            if self.mode == 'val':
                patient_name = self.val_names[new_idx]
            if self.mode == 'train':
                patient_name = self.train_names[new_idx]
            
            
            
        all_actions = torch.tensor(np.array(patient_info['all_actions']))
        all_obs = torch.tensor(np.array(patient_info['all_obs']))
        
        # from prostate mask, obtain max depth
        read_img = ImageReader()    
        prostate_mask = np.array(np.transpose((read_img(os.path.join(self.prostate_folder, patient_name))), [1, 2, 0]))
        max_depth = np.max(np.where(prostate_mask == 1)[-1])
    
        # 2. Randomly sample time idx (from t = 0:T)
        NUM_STEPS = all_obs.size()[0]
        sampled_idx = np.random.randint(0,NUM_STEPS)
        obs = all_obs[sampled_idx, :,:,:]
        action = all_actions[sampled_idx, :]    

        # 3. Return torch.tensor(obs), torch.tesnor(actions)
        prostate_centroid = torch.tensor(np.array(patient_info['prostate_centroid']))
        lesion_centroid = torch.tensor(np.array(patient_info['tumour_centroid']))
        lesion_mask = torch.tensor(np.array(patient_info['lesion_mask']))
        
        return obs, action, prostate_centroid, lesion_centroid, lesion_mask, max_depth
    
    def __len__(self):
        return self.dataset_len[self.mode]
    
def separate_actions(comb_actions, all_ids, idx):
    """
    A function that obtains actions only corresponding to lesion with idx 

    :comb_actions: 3xN action array 
    :all_ids: 3x1 action identifiers; ids for which lesion each action corresponds to 

    Returns:
    :sep_actions: 3 x num_steps where num_steps is the number of steps corresponding to chosen idx lesion 
    """


    idx_map = (all_ids == idx)[0]

    # return actions corresponding to lesion idx only 
    comb_actions = np.transpose(comb_actions)
    sep_actions = comb_actions[:,idx_map]
    sep_actions = np.transpose(sep_actions)

    return sep_actions 

def separate_masks(multiple_lesions, lesion_idx):

    # for each lesion: 
    unique_idx = np.unique(multiple_lesions)
    num_lesion_masks = len(unique_idx) - 1
    ds_lesion_mask = multiple_lesions[::2, ::2, ::4]
    row, col, depth = np.shape(ds_lesion_mask)

    img_vol = np.zeros((row, col, depth))

    for i in range(depth):
        img_vol[:,:,i] = (ds_lesion_mask[:,:,i] == lesion_idx)

    return img_vol 



    #for idx in range(num_lesion_masks):
    #    mask = multiple_lesions[(multiple_lesions == idx+1)] # add 1 as we don't want background 
    #    all_masks.append(mask)

    return all_masks 

def find_value_changes(arr):
    diff = np.diff(arr)  # Calculate the difference between consecutive elements
    change_indices = np.where(diff != 0)[1]  # Find the indices where the difference is nonzero
    
    return change_indices

class TimeStep_data_steps(Dataset):

    def __init__(self, folder_name, csv_path= 'csv_file.csv', labels_path = 'action_labels.h5', mode = 'train', step = 'c2l', single_patient = False, T = 3):

        self.folder_name = folder_name
        self.mode = mode
        self.labels_path = labels_path
        self.step = step # string only 'c2l', 'n2n', 'l2l' 
        self.single_patient = single_patient
        self.T = T # number of timesteps to use for channel 
        
        # Obtain list of patient names with multiple lesions -> change to path name
        #df_dataset = pd.read_csv('/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv')
        #z3df_dataset = pd.read_csv('/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv')
        df_dataset = pd.read_csv(csv_path)
        
        #Filter out patients >=5 lesions 
        patients_w5 = np.where(df_dataset[' num_lesions'] >= 5)[0] # save these indices for next time!!!
        # Remove patients where lesions >5 as these are incorrectly labelled!!
        df_dataset = df_dataset.drop(df_dataset.index[patients_w5])
    
        self.all_file_names = df_dataset['patient_name'].tolist()
        self.num_lesions = df_dataset[' num_lesions'].tolist()
        
        self.vol_creater = GridArray() 

        # read h5 file 
        self.grid_labels = h5py.File(labels_path, 'r')
        size_dataset = len(self.all_file_names)
        train_len = int(size_dataset * 0.7) 
        test_len = int(size_dataset * 0.2) 
        val_len = size_dataset - (train_len + test_len)

        # both test and val have simila rnumber of lesions (mean = 2.4 lesions)
        self.train_names = self.all_file_names[0:train_len]
        self.val_names = self.all_file_names[train_len:train_len + val_len]
        self.test_names = self.all_file_names[train_len + val_len:]
        self.dataset_len = {'train' : train_len, 'test': test_len, 'val' : val_len}

        # Folder names
        self.lesion_folder = os.path.join(folder_name, 'lesion')
        self.mri_folder = os.path.join(folder_name, 't2w')
        self.prostate_folder = os.path.join(folder_name, 'prostate_mask')

    def _normalise(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = np.max(img)
        min_img = np.min(img)

        #Normalise values between 0 to 1
        normalised_img =  ((img - min_img)/(max_img - min_img)) 

        return normalised_img.astype(np.float32)
    
    def _normalise_actions(self, actions):

        normalised_actions = np.zeros_like(actions)

        # First two actions x,y in delta range (-2,2)
        normalised_actions[0:2] = actions[0:2] / 2
        #normalised_actions[-1] = (actions[-1] - 0.5) *2 
        normalised_actions[-1] = actions[2] - 1

        return normalised_actions 

    def __len__(self):
        return self.dataset_len[self.mode]
 
    def __getitem__(self, idx):
        """
        Gets items from train, test, val 

        # From each patient idx 
            # randomly sample 3 sequential images starting from i = 1 
            # randomly sample 3 images 

        Returns:
        ------------
        template_grid : 100 x 100 x 25 (dataset)
        actions: 
        """

        if self.mode == 'train':
            #idx_ = idx
            if self.single_patient: 
                idx = 1
            patient_name = self.train_names[idx]

        elif self.mode == 'val':
            #idx_ = idx + self.dataset_len['train']
            patient_name = self.val_names[idx]

        elif self.mode == 'test':
            #idx_ = idx + self.dataset_len['train'] + self.dataset_len['val']
            patient_name = self.test_names[idx]

        # Read prostate mask, lesion mask, prostate mask separately using ImageReader    
        #patient_name = self.all_file_names[idx_]
        read_img = ImageReader()    
        mri_vol = np.transpose(self._normalise(read_img(os.path.join(self.mri_folder, patient_name))), [1, 2, 0])
        lesion_mask = np.transpose(self._normalise(read_img(os.path.join(self.lesion_folder, patient_name))), [1, 2, 0])
        prostate_mask = np.transpose(self._normalise(read_img(os.path.join(self.prostate_folder, patient_name))), [1, 2, 0])
        
        # downsampled prostate mask 
        ds_prostate_mask = prostate_mask[::2,::2,::4]
        max_depth = np.max(np.where(ds_prostate_mask == 1)[-1])
        ds_lesion_mask = lesion_mask[::2,::2,::4]

        # Obtain combined prostate_lesion and turn into torch tensor
        combined_mask = (torch.from_numpy(self.get_img_mask(prostate_mask, lesion_mask))[0::2, 0::2, 0::4])/2
        
        # Get file path name 
        sitk_img_path = os.path.join(self.lesion_folder, patient_name)

        # Read grid image label 
        patient_file = self.grid_labels[patient_name]
        all_actions = np.array(patient_file['all_actions'])
        #all_grids = np.array(patient_file['all_grids'])
        all_pos = np.array(patient_file['current_pos'])
        #all_obs_ids= np.array(patient_file['obs_identifiers'])

        # obtain lesion mask: 
        multiple_masks = patient_file['multiple_lesion_img']
        NUM_LESIONS = len(np.unique(multiple_masks)) - 1 # remove background as 1
        all_ids = np.array(patient_file['action_identifiers'])
        
        # Order lesion sizes in index 
        lesion_size_idx = find_unique_values(all_ids)
        
        #print('chicken')
        
        if (self.step == 'c2l') or (self.step == 'n2n'): #c2l or n2n 

            # sample lesion 
            lesion_idx = np.random.choice((np.arange(1, NUM_LESIONS)))
            #lesion_mask = separate_masks(multiple_masks, lesion_idx)
            lesion_mask = 1*(np.array(multiple_masks) == lesion_idx)[::2,::2,::4]
            
            # resample if empty
            empty_mask = (len(np.unique(lesion_mask)) == 1)
            
            while empty_mask:
                print(f"Empty mask")
                lesion_idx = np.random.choice((np.arange(1, NUM_LESIONS)))
                #lesion_mask = separate_masks(multiple_masks, lesion_idx)
                lesion_mask = 1*(np.array(multiple_masks) == lesion_idx)[::2,::2,::4]
                
                # Checking if still empty 
                empty_mask = (len(np.unique(lesion_mask)) == 1)
                

            # separate lesion actions and current grid positions 
            lesion_actions = separate_actions(all_actions, all_ids, lesion_idx-1)
            lesion_positions = separate_actions(all_pos, all_ids, lesion_idx-1)
            NUM_ACTIONS_LESION = np.shape(lesion_actions)[0]

            # needle_vol = []
            # for pos_idx in range(NUM_ACTIONS_LESION):
            #     current_pos = lesion_positions[pos_idx,:]
            #     needle_vol.append(self.vol_creater.create_needle_vol(current_pos, max_depth))

            if self.step == 'c2l': #centre to each lesion -> start from first action only, ie action_idx = 0 
                random_idx = 0 
            elif self.step == 'n2n':
                random_idx = np.random.choice(np.arange(1, NUM_ACTIONS_LESION))
            else:
                random_idx = np.random.choice(np.arange(0, NUM_ACTIONS_LESION))
                
            ### OBTAIN ACTIONS 
            final_action = lesion_actions[random_idx,:] # Only consider final action to be estimated
            
            ### OBTAIN OBSERVATIONS 

            # Fixed from time step T - 2 : T instead of T : T+2
            needle_stack = self.get_obs(random_idx, lesion_positions, max_depth)
            
            # all_grid = torch.zeros((100,100,24))
            # # for lesion_idx in range(1,NUM_LESIONS+1):
            # #     lesion_positions = separate_actions(all_pos, all_obs_ids, lesion_idx-1)
            # #     lesion_actions = separate_actions(all_actions, all_ids, lesion_idx-1)
            # print(f"lesion_idx : {lesion_idx}")
            # for i in range(len(lesion_actions)):
            #     all_grid += self.vol_creater.create_needle_vol(lesion_positions[i, :], max_depth)
            # plt.figure()
            # #plt.imshow(np.max(all_grid.numpy(), axis =2)*10 + np.max(lesion_mask, axis = 2))
            # plt.imshow(np.max(all_grid.numpy(), axis =2)*10 + np.max(multiple_masks[::2,::2,::4], axis = 2))
            # if random_idx == 0:
            #     #sampled_grid = np.array([np.zeros((100,100)), np.zeros((100,100)), np.array(all_grids[random_idx])])
            #     # start array : 
            #     needle_stack = torch.tensor(np.array([np.zeros([100,100,24]), np.zeros([100,100,24]), needle_vol[random_idx]]))
            # elif random_idx == 1:
            #     #sampled_grid = np.array([np.zeros((100,100)), np.array(all_grids[random_idx-1]), np.array(all_grids[random_idx])])
            #     needle_stack = torch.tensor(np.array([np.zeros([100,100,24]), needle_vol[random_idx-1], needle_vol[random_idx]]))
            # else:
            #     #sampled_grid = all_grids[random_idx - 2 :random_idx+1 , :]
            #     needle_stack = torch.tensor(np.array(needle_vol[random_idx-2 : random_idx+1]))
                
            # Combined grid : Contains template grid pos chosen at time steps t-3 : T
            # Final action : Action to take from time step T 
            final_action = torch.tensor(self._normalise_actions(final_action))#, axis = 0)

            #TODO : test this!!! Additional informaiton for metrics computation 
            downsampled_lesion_mask = lesion_mask
            lesion_centroids = np.array(patient_file['lesion_centroids']) # lesion centroids for computing distance metric
            action_identifier = lesion_idx 
            #all_identifiers = np.array(patient_file['action_identifiers']) # index of lesions to visit 
            #action_identifier = int(all_identifiers[0,random_idx])
            tumour_centroid = lesion_centroids[action_identifier,:]

            ## Obtain observation of needle volumes: 
            obs_stack = torch.cat((torch.tensor(lesion_mask).unsqueeze(0), torch.tensor(ds_prostate_mask).unsqueeze(0), needle_stack), axis = 0)

            # Obtain actions and current position at time step chosen 
            grid_pos = all_pos[random_idx, :]

        elif self.step == 'wacky':
            
            all_obs_ids= np.array(patient_file['obs_identifiers'])
            
            # sample lesion 
            lesion_idx = np.random.choice((np.arange(1, NUM_LESIONS)))
            #lesion_mask = separate_masks(multiple_masks, lesion_idx)
            lesion_mask = 1*(np.array(multiple_masks) == lesion_idx)[::2,::2,::4]
            
            # resample if empty
            empty_mask = (len(np.unique(lesion_mask)) == 1)
            
            # KEEP SAMPLING UNTIL EMPTY NON LESION 
            while empty_mask:
                
                print(f"Empty mask -> choosing again")
                lesion_idx = np.random.choice((np.arange(1, NUM_LESIONS)))
                lesion_mask = 1*(np.array(multiple_masks) == lesion_idx)[::2,::2,::4]
                
                empty_mask = (len(np.unique(lesion_mask)) == 1)
                
            # separate lesion actions and current grid positions 
            lesion_actions = separate_actions(all_actions, all_ids, lesion_idx-1)
            lesion_positions = separate_actions(all_pos, all_obs_ids, lesion_idx-1)
            NUM_ACTIONS_LESION = np.shape(lesion_actions)[0]
        
            ### OBTAIN ACTIONS 
            random_idx = np.random.choice(np.arange(0, NUM_ACTIONS_LESION))
            final_action = lesion_actions[random_idx,:] # Only consider final action to be estimated
            
            ### OBTAIN OBSERVATIONS 

            # Fixed from time step T - 2 : T instead of T : T+2
            needle_stack = self.get_obs_wacky(random_idx, lesion_positions, max_depth)
            #plt.figure()
            #plt.imshow(np.max(needle_stack[2,:,:,:].numpy(), axis = 2) + np.max(lesion_mask, axis = 2))
            
            # for lesion_idx in range(1,NUM_LESIONS+1):
            #     lesion_positions = separate_actions(all_pos, all_obs_ids, lesion_idx-1)
            #     lesion_actions = separate_actions(all_actions, all_ids, lesion_idx-1)
            
            # all_grid = torch.zeros((100,100,24))
            # for i in range(len(lesion_actions)+1):
            #     i = random_idx
            #     print(lesion_positions[i, :])
            #     all_grid += self.vol_creater.create_needle_vol_wacky(lesion_positions[i, :], max_depth)
            # plt.figure()
            # #plt.imshow(np.max(all_grid.numpy(), axis =2)*10 + np.max(lesion_mask, axis = 2))
            # plt.imshow(np.max(all_grid.numpy(), axis =2)*5 + np.max(multiple_masks[::2,::2,::4], axis = 2))
            
            # if random_idx == 0:
            #     #sampled_grid = np.array([np.zeros((100,100)), np.zeros((100,100)), np.array(all_grids[random_idx])])
            #     # start array : 
            #     needle_stack = torch.tensor(np.array([np.zeros([100,100,24]), np.zeros([100,100,24]), needle_vol[random_idx]]))
            # elif random_idx == 1:
            #     #sampled_grid = np.array([np.zeros((100,100)), np.array(all_grids[random_idx-1]), np.array(all_grids[random_idx])])
            #     needle_stack = torch.tensor(np.array([np.zeros([100,100,24]), needle_vol[random_idx-1], needle_vol[random_idx]]))
            # else:
            #     #sampled_grid = all_grids[random_idx - 2 :random_idx+1 , :]
            #     needle_stack = torch.tensor(np.array(needle_vol[random_idx-2 : random_idx+1]))
                
            # Combined grid : Contains template grid pos chosen at time steps t-3 : T
            # Final action : Action to take from time step T 
            
            final_action = torch.tensor(self._normalise_actions(final_action))#, axis = 0)

            #TODO : test this!!! Additional informaiton for metrics computation 
            downsampled_lesion_mask = lesion_mask
            lesion_centroids = np.array(patient_file['lesion_centroids']) # lesion centroids for computing distance metric
            action_identifier = lesion_idx 
            all_identifiers = np.array(patient_file['action_identifiers']) # index of lesions to visit 
            action_identifier = int(all_identifiers[0,random_idx])
            tumour_centroid = lesion_centroids[action_identifier,:]

            ## Obtain observation of needle volumes: 
            obs_stack = torch.cat((torch.tensor(lesion_mask).unsqueeze(0), torch.tensor(ds_prostate_mask).unsqueeze(0), needle_stack), axis = 0)

            # Obtain actions and current position at time step chosen 
            grid_pos = all_pos[random_idx, :]
            
        elif self.step == 'e2e': # end to end training, sample any random index 

            # Down sample for refernece 
            lesion_mask = lesion_mask[::2,::2,::4]

            # sample random index for actions and observations 
            NUM_ACTIONS = np.shape(all_actions)[0]
            random_idx = np.random.choice(np.arange(0, NUM_ACTIONS-1))

            # Obtain action 
            final_action = all_actions[random_idx,:] # Only consider final action to be estimated
            final_action = torch.tensor(self._normalise_actions(final_action))#, axis = 0)

            # Obtain observations from T = t-2 to T = 0 
            
            # # First obtain all observaitons 
            # needle_vol = []
            # for pos_idx in range(NUM_ACTIONS):
            #     current_pos = all_pos[pos_idx,:]
            #     needle_vol.append(self.vol_creater.create_needle_vol(current_pos, max_depth))
            
            needle_stack = self.get_obs(random_idx, all_pos, max_depth)
            
            downsampled_lesion_mask = lesion_mask
            lesion_centroids = np.array(patient_file['lesion_centroids']) # lesion centroids for computing distance metric
            action_identifier = int(all_ids[0][random_idx])
            tumour_centroid = lesion_centroids[action_identifier,:] # moving towards l2 

            ## Obtain observation of needle volumes: 
            obs_stack = torch.cat((torch.tensor(lesion_mask).unsqueeze(0), torch.tensor(ds_prostate_mask).unsqueeze(0), needle_stack), axis = 0)

            # Obtain actions and current position at time step chosen 
            grid_pos = all_pos[random_idx, :]
        
        elif self.step == 'wacky_multiple': # end to end training, sample any random index 

            # Down sample for refernece 
            lesion_mask = lesion_mask[::2,::2,::4]

            # sample random index for actions and observations 
            NUM_ACTIONS = np.shape(all_actions)[0]
            random_idx = np.random.choice(np.arange(0, NUM_ACTIONS-1))

            # Obtain action 
            final_action = all_actions[random_idx,:] # Only consider final action to be estimated
            final_action = torch.tensor(self._normalise_actions(final_action))#, axis = 0)

            # Obtain observations from T = t-2 to T = 0 
            
            # # First obtain all observaitons 
            # needle_vol = []
            # for pos_idx in range(NUM_ACTIONS):
            #     current_pos = all_pos[pos_idx,:]
            #     needle_vol.append(self.vol_creater.create_needle_vol(current_pos, max_depth))
            
            needle_stack = self.get_obs_wacky(random_idx, all_pos, max_depth)
            
            downsampled_lesion_mask = lesion_mask
            lesion_centroids = np.array(patient_file['lesion_centroids']) # lesion centroids for computing distance metric
            action_identifier = int(all_ids[0][random_idx])
            tumour_centroid = lesion_centroids[action_identifier,:] # moving towards l2 
            
            # tumour needle stack 
            
            ## Obtain observation of needle volumes: 
            lesion_masks = torch.zeros((4, 200, 200, 96))
            #print(f"Number of lesions {self.num_lesions}")
        
            for i, tum_idx in enumerate(lesion_size_idx):
            
                # Should account for num_lesions <= 4 
                #print(f'Number : {i} tum_idx :{tum_idx}')
                #lesion_masks[i,:,:,:] = separate_masks(multiple_masks, tum_idx)
                lesion_masks[i,:,:,:] = torch.tensor(np.array(multiple_masks) == (int(tum_idx) +1))
            
            # Downsample them 
            lesion_masks = lesion_masks[:, ::2,::2,::4]
            
            ## Obtain observation of needle volumes: 
            obs_stack = torch.cat((lesion_masks, torch.tensor(ds_prostate_mask).unsqueeze(0), needle_stack), axis = 0)

            # Obtain actions and current position at time step chosen 
            grid_pos = all_pos[random_idx, :]
             
        else: # l2l : sample transition 
            
            # sample transitions 
            all_transitions = find_value_changes(all_ids) # find all idx where transiiton occurs 
            transition_idx = int(np.random.choice(all_transitions))  # sample a transition

            # sampled lesions 
            l1 = int(all_ids[0][transition_idx])
            l2 = int(all_ids[0][transition_idx+1]) 
            action_identifier = l2 

            # obtain observations
            l1_mask = separate_masks(multiple_masks, l1+1) # add 1 as lesions are 1-indexed; 0 is background 
            l2_mask = separate_masks(multiple_masks, l2+1)
            lesion_mask = l1_mask + l2_mask
            
            # separate lesion actions and current grid positions 

            # sample actions for L2 => ie want first action leading to next lesion 
            lesion_actions = separate_actions(all_actions, all_ids, l2) # actions for l2 
            final_action = lesion_actions[0,:]
            final_action = torch.tensor(self._normalise_actions(final_action))#, axis = 0)

            # sample observaitons for L1 : ie observaitons leading up to next action 
            lesion_positions = separate_actions(all_pos, all_ids, l1) # observations for l1 
            NUM_ACTIONS_LESION = np.shape(lesion_positions)[0]

            needle_vol = []
            for pos_idx in range(NUM_ACTIONS_LESION):
                current_pos = lesion_positions[pos_idx,:]
                needle_vol.append(self.vol_creater.create_needle_vol(current_pos, max_depth))

            # Obtain needle stack from T-2, T-1 and T 
            needle_stack = torch.tensor(np.array(needle_vol[-3:]))

            #TODO : test this!!! Additional informaiton for metrics computation 
            downsampled_lesion_mask = lesion_mask
            lesion_centroids = np.array(patient_file['lesion_centroids']) # lesion centroids for computing distance metric
            tumour_centroid = lesion_centroids[l2,:] # moving towards l2 
                
            obs_stack = torch.cat((torch.tensor(lesion_mask).unsqueeze(0), torch.tensor(ds_prostate_mask).unsqueeze(0), needle_stack), axis = 0)

            # Obtain actions and current position at time step chosen 
            grid_pos = lesion_positions[-1, :]

        return obs_stack, final_action, downsampled_lesion_mask, action_identifier, tumour_centroid, grid_pos, max_depth 
    
    def get_img_mask(self, prostate_mask, tumour_mask):
        """
        Adds up the prostate and tumour mask together to obtain binary masks of these two objects together
        """

        prostate_vol = prostate_mask[:, :, :] #prostate = 1
        tumour_vol = tumour_mask[:, :, :] * 2 #tumour = 2
        combined_tumour_prostate = prostate_vol + tumour_vol

        # if lesion and prostate intersection, keep us lesion label 
        combined_tumour_prostate[combined_tumour_prostate >= 2] = 2

        return combined_tumour_prostate 

    def get_obs_wacky(self, random_idx, all_pos, max_depth):
        """
        Creates observation based on number of channels T to use 
        
        Uses create_vol_wacky which says x = action[0] and y = action[1]
        whilst create_vol says x = action[1] and y = action[0]

        Args:
            random_idx (float): _description_
            all_pos (ndarray) : all current positions for lesion 
            max_depth (float) : max depth of prostate gland, to use for plotting volumes 
        """
        
        needle_stack = torch.zeros([self.T, 100, 100, 24])
        #random_idx = 5
        
        if random_idx == 0:
            
            # most recent needle position 
            needle_stack[-1, :, :, :] = torch.tensor(self.vol_creater.create_needle_vol_wacky(all_pos[random_idx, :], max_depth))
            
            #sampled_grid = np.array([np.zeros((100,100)), np.zeros((100,100)), np.array(all_grids[random_idx])])
            #needle_vol = self.vol_creater.create_needle_vol(all_pos[random_idx, :], max_depth)
            #needle_stack = torch.tensor(np.array([np.zeros([100,100,24]), np.zeros([100,100,24]), needle_vol]))
        
        elif random_idx == 1:
            
            needle_stack[-1, :, :, :] = torch.tensor(self.vol_creater.create_needle_vol_wacky(all_pos[random_idx, :], max_depth))
            needle_stack[-2, :, :, :] = torch.tensor(self.vol_creater.create_needle_vol_wacky(all_pos[random_idx-1, :], max_depth))
            
            #sampled_grid = np.array([np.zeros((100,100)), np.array(all_grids[random_idx-1]), np.array(all_grids[random_idx])])
            #needle_vol_t_1 = self.vol_creater.create_needle_vol(all_pos[random_idx-1, :], max_depth)
            #needle_vol_t0 = self.vol_creater.create_needle_vol(all_pos[random_idx, :], max_depth)
            #needle_stack = torch.tensor(np.array([np.zeros([100,100,24]), needle_vol_t_1, needle_vol_t0]))
        
        else:
            #print(f"random_idx {random_idx}")
            idx = 0 
            for i in range((random_idx+1 - self.T), random_idx +1):
                #print(f" image idx : {i} idx : {idx} ")
                #print(f"all_pos {all_pos[i, :]}")
                needle_stack[idx, :,:,:] = torch.tensor(self.vol_creater.create_needle_vol_wacky(all_pos[i, :], max_depth))
                idx += 1
            
            #needle_vol_t_2 = self.vol_creater.create_needle_vol(all_pos[random_idx-2, :], max_depth)
            #needle_vol_t_1 = self.vol_creater.create_needle_vol(all_pos[random_idx-1, :], max_depth)
            #needle_vol_t0 = self.vol_creater.create_needle_vol(all_pos[random_idx, :], max_depth)
            #needle_stack = torch.tensor([needle_vol_t_2, needle_vol_t_1, needle_vol_t0])
        
        return needle_stack 

    def get_obs(self, random_idx, all_pos, max_depth):
        """
        Creates observation based on number of channels T to use 

        Args:
            random_idx (float): _description_
            all_pos (ndarray) : all current positions for lesion 
            max_depth (float) : max depth of prostate gland, to use for plotting volumes 
        """
        
        needle_stack = torch.zeros([self.T, 100, 100, 24])
        #random_idx = 5
        
        if random_idx == 0:
            
            # most recent needle position 
            needle_stack[-1, :, :, :] = torch.tensor(self.vol_creater.create_needle_vol(all_pos[random_idx, :], max_depth))
            
            #sampled_grid = np.array([np.zeros((100,100)), np.zeros((100,100)), np.array(all_grids[random_idx])])
            #needle_vol = self.vol_creater.create_needle_vol(all_pos[random_idx, :], max_depth)
            #needle_stack = torch.tensor(np.array([np.zeros([100,100,24]), np.zeros([100,100,24]), needle_vol]))
        
        elif random_idx == 1:
            
            needle_stack[-1, :, :, :] = torch.tensor(self.vol_creater.create_needle_vol(all_pos[random_idx, :], max_depth))
            needle_stack[-2, :, :, :] = torch.tensor(self.vol_creater.create_needle_vol(all_pos[random_idx-1, :], max_depth))
            
            #sampled_grid = np.array([np.zeros((100,100)), np.array(all_grids[random_idx-1]), np.array(all_grids[random_idx])])
            #needle_vol_t_1 = self.vol_creater.create_needle_vol(all_pos[random_idx-1, :], max_depth)
            #needle_vol_t0 = self.vol_creater.create_needle_vol(all_pos[random_idx, :], max_depth)
            #needle_stack = torch.tensor(np.array([np.zeros([100,100,24]), needle_vol_t_1, needle_vol_t0]))
        
        else:
            
            idx = 0 
            for i in range((random_idx+1 - self.T), random_idx +1):
                #print(f" image idx : {i} idx : {idx} ")
                needle_stack[idx, :,:,:] = torch.tensor(self.vol_creater.create_needle_vol(all_pos[i, :], max_depth))
                idx += 1
            
            #needle_vol_t_2 = self.vol_creater.create_needle_vol(all_pos[random_idx-2, :], max_depth)
            #needle_vol_t_1 = self.vol_creater.create_needle_vol(all_pos[random_idx-1, :], max_depth)
            #needle_vol_t0 = self.vol_creater.create_needle_vol(all_pos[random_idx, :], max_depth)
            #needle_stack = torch.tensor([needle_vol_t_2, needle_vol_t_1, needle_vol_t0])
        
        return needle_stack 
         
class TimeStep_data_debug(Dataset):
    """
    Debugging dataloader : always returns the same action [1 0 0] ie always going in a horizontal striaghtg line -> 
    """

    def __init__(self, folder_name, labels_path = 'action_labels.h5', mode = 'train', finetune = False):

        self.folder_name = folder_name
        self.mode = mode
        self.labels_path = labels_path
        self.finetune = finetune

        # Obtain list of patient names with multiple lesions -> change to path name
        df_dataset = pd.read_csv('/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv')
        #df_dataset = pd.read_csv('/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv')
        self.all_file_names = df_dataset['patient_name'].tolist()
        self.num_lesions = df_dataset[' num_lesions'].tolist()

        # read h5 file 
        self.grid_labels = h5py.File(labels_path, 'r')
        size_dataset = len(self.all_file_names)
        train_len = int(size_dataset * 0.7) 
        test_len = int(size_dataset * 0.2) 
        val_len = size_dataset - (train_len + test_len)

        # both test and val have simila rnumber of lesions (mean = 2.4 lesions)
        self.train_names = self.all_file_names[0:train_len]
        self.val_names = self.all_file_names[train_len:train_len + val_len]
        self.test_names = self.all_file_names[train_len + val_len:]
        self.dataset_len = {'train' : train_len, 'test': test_len, 'val' : val_len}

        # Folder names
        self.lesion_folder = os.path.join(folder_name, 'lesion')
        self.mri_folder = os.path.join(folder_name, 't2w')
        self.prostate_folder = os.path.join(folder_name, 'prostate_mask')

    def _normalise(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255
        """

        max_img = np.max(img)
        min_img = np.min(img)

        #Normalise values between 0 to 1
        normalised_img =  ((img - min_img)/(max_img - min_img)) 

        return normalised_img.astype(np.float32)
    
    def _normalise_actions(self, actions):

        normalised_actions = np.zeros_like(actions)

        # First two actions x,y in delta range (-2,2)
        normalised_actions[0:2] = actions[0:2] / 2
        normalised_actions[-1] = (actions[-1] - 0.5) *2 

        return normalised_actions 

    def __len__(self):
        return self.dataset_len[self.mode]
 
    def __getitem__(self, idx):
        """
        Gets items from train, test, val 

        # From each patient idx 
            # randomly sample 3 sequential images starting from i = 1 
            # randomly sample 3 images 

        Returns:
        ------------
        template_grid : 100 x 100 x 25 (dataset)
        actions: 
        """

        if self.mode == 'train':
            #idx_ = idx
            patient_name = self.train_names[idx]

        elif self.mode == 'val':
            #idx_ = idx + self.dataset_len['train']
            patient_name = self.val_names[idx]

        elif self.mode == 'test':
            #idx_ = idx + self.dataset_len['train'] + self.dataset_len['val']
            patient_name = self.test_names[idx]


        # Read prostate mask, lesion mask, prostate mask separately using ImageReader    
        #patient_name = self.all_file_names[idx_]
        read_img = ImageReader()    
        mri_vol = np.transpose(self._normalise(read_img(os.path.join(self.mri_folder, patient_name))), [1, 2, 0])
        lesion_mask = np.transpose(self._normalise(read_img(os.path.join(self.lesion_folder, patient_name))), [1, 2, 0])
        prostate_mask = np.transpose(self._normalise(read_img(os.path.join(self.prostate_folder, patient_name))), [1, 2, 0])
        
        # Obtain combined prostate_lesion and turn into torch tensor
        combined_mask = (torch.from_numpy(self.get_img_mask(prostate_mask, lesion_mask))[0::2, 0::2, 0::4])/2
        
        # Get file path name 
        sitk_img_path = os.path.join(self.lesion_folder, patient_name)

        # Read grid image label 
        patient_file = self.grid_labels[patient_name]
        all_actions = np.array(patient_file['all_actions'])
        all_grids = np.array(patient_file['all_grids'])
        NUM_ACTIONS = np.shape(all_actions)[0]
        
        # Randomly sample idx from 0: T-3 (num_actions-2 because anrage aranges from 0 to n-1 val)
        if self.finetune: 
            random_idx = 0 # finetune only with first frames, to let agent know how to start the actions with different datasets!!! 
        else:
            random_idx = np.random.choice(np.arange(0, NUM_ACTIONS-1))
        
        #if self.mode == 'val':
        print(f"Patient_idx : {idx} action_idx : {random_idx}")

        #print(f'Len of actions : {NUM_ACTIONS} random idx : {random_idx}')
        #sampled_actions = all_actions[random_idx:random_idx+3, :]
        final_action = all_actions[random_idx,:] # Only consider final action to be estimated
        
        # Fixed from time step T - 2 : T instead of T : T+2
        if random_idx == 0:
            sampled_grid = np.array([np.zeros((100,100)), np.zeros((100,100)), np.array(all_grids[random_idx])])
        elif random_idx == 1:
            sampled_grid = np.array([np.zeros((100,100)), np.array(all_grids[random_idx-1]), np.array(all_grids[random_idx])])
        else:
            sampled_grid = all_grids[random_idx - 2 :random_idx+1 , :]
            
        combined_grid = np.zeros((100,100,75))
        
        for i in range(3): 
            idx = i*25
            combined_grid[:,:,idx] = sampled_grid[i,:,:]
            combined_grid[:,:,idx+1: (i+1)*25] = combined_mask 
            #print(f'Idx : {idx}')
            #print(f' Idx : {idx+1} : {(i+1)*25}')

        # Combined grid : Contains template grid pos chosen at time steps t-3 : T
        # Final action : Action to take from time step T 
        combined_grid = torch.unsqueeze(torch.tensor(combined_grid), axis = 0)
        #final_action = torch.tensor(self._normalise_actions(final_action))#, axis = 0)

        # debugging action : [1 0 0]
        final_action = torch.tensor((1, 0 ,0)) 


        # Combined actions 
        
        return combined_grid, final_action  
    
    def get_img_mask(self, prostate_mask, tumour_mask):
        """
        Adds up the prostate and tumour mask together to obtain binary masks of these two objects together
        """

        prostate_vol = prostate_mask[:, :, :] #prostate = 1
        tumour_vol = tumour_mask[:, :, :] * 2 #tumour = 2
        combined_tumour_prostate = prostate_vol + tumour_vol

        # if lesion and prostate intersection, keep us lesion label 
        combined_tumour_prostate[combined_tumour_prostate >= 2] = 2

        return combined_tumour_prostate 

class SimpleFeatureExtractor_3D_continuous(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        Correspods to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, multiple_frames = False, num_multiple_frames = 3):
        
        super(SimpleFeatureExtractor_3D_continuous, self).__init__(observation_space, features_dim)
        # Assumes CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        num_input_channels = observation_space.shape[-1] #rows x cols x channels 
        #num_multiple_frames = 3
        num_multiple_frames = num_multiple_frames
        self.num_multiple_frames = num_multiple_frames
        self.cnn_layers = nn.Sequential(

            # First layer like resnet, stride = 2
            nn.Conv3d(num_multiple_frames, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            # Apply pooling layer in between 
            nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            # Apply pooling layer in between 
            nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1),

            nn.Conv3d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        #Flatten layers 
        self.flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            all_layers = nn.Sequential(self.cnn_layers, self.flatten)
            
            #observation_space_shuffled = np.transpose(observation_space.sample(), [2, 1, 0])
            #n_flatten = all_layers(torch.as_tensor(observation_space_shuffled[None]).float()).shape[1]
            processed_obs_space = self._pre_process_image(torch.as_tensor((observation_space.sample()[None]))).float()
            n_flatten = all_layers(processed_obs_space).shape[1]  

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        observations = self._pre_process_image(observations)
        observations = observations.float() 
        output = self.cnn_layers(observations)
        output = self.flatten(output)
        
        return self.linear(output)

    def _pre_process_image(self, images):
        """ 
        A function that switches the dimension of image from row x col x channel -> channel x row x colmn 
        and addeds a dimension along 0th axis to fit network 
        """ 
        #print(f'Image size {images.size()}')
        image = images.clone().detach().to(torch.uint8)#.squeeze()
        if len(np.shape(images)) == 5:
            image = image.squeeze()
        split_channel_image = torch.cat([torch.cat([image[j,:,:,i*25:(i*25)+25].unsqueeze(0) for i in range(3)]).unsqueeze(0) for j in range(image.size()[0])])#.clone().detach().to(torch.uint8)
        #split_channel_image = torch.cat([torch.cat(torch.tensor_split(image[i,:,:,:].unsqueeze(0), self.num_multiple_frames, dim=3)).unsqueeze(0) for i in range(image.size()[0])])
        #processed_image = image.permute(0, 3,2,1)
        #processed_image = torch.unsqueeze(processed_image, dim= 0)
        
        # Turn image from channel x row x column -> channel x row x column x depth for pre-processing with 3D layers 

        return split_channel_image

def compute_rmse(mse):
    """
    Computes rmse as a metric to be reported, but not to be used for comptuation of loss fns 
    """

    with torch.no_grad():
        mse_copy = torch.clone(mse.detach())
        rmse = torch.sqrt(mse_copy)

    return rmse 

# Train scripts using old observation : 3 x 100 x 100 x 25 where 3 is num channels, where each timestep obs is a channel 
def train_pertimestep(model, agent, train_dataloader, val_dataloader, num_epochs = 10, use_cuda = False, save_folder = 'model_1', loss_fn_str = 'MSE', clip_actions = False):
    
    """
    A function that performs the training and validation loop 
    :params:
    :prostate_dataloader: torch dataloader that can load images and masks in
    :num_epochs: (int) Number of epochs to train for 
    """
    current_dir = os.getcwd()
    train_folder = os.path.join(current_dir, save_folder)

    os.makedirs(train_folder, exist_ok = True) 
    print(f'train folder path : {train_folder}')
    writer = SummaryWriter(os.path.join(train_folder, 'runs')) 

    if use_cuda:
        model.cuda()
    
    # Defining optimiser, loss fn 
    #loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.MSELoss()
    print('loss fn : MSE Loss')

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-04)

    # Parameters 
    step = 0 
    freq_print = 4
    freq_eval = 4

    # Saving files 
    all_loss_train = np.zeros((num_epochs,1))
    all_acc_train = np.zeros((num_epochs, 1))
    all_loss_val = [] 
    all_acc_val = []
    best_loss = np.inf 
    best_acc = 0 
    
    csv_train_path = os.path.join(train_folder, 'train_loss.csv')
    csv_val_path = os.path.join(train_folder, 'val_loss.csv')

    with open(csv_train_path, 'w') as fp:
        fp.write('''epoch_num, bce_loss''')
        fp.write('\n')

    with open(csv_val_path, 'w') as fp:
        fp.write('''epoch_num, bce_loss''')
        fp.write('\n')

    
    for epoch_no in range(num_epochs):
        
        print(f"\n Epoch number : {epoch_no}")
        acc_vals = []
        loss_vals = [] 
        hr_vals = [] 
        ccl_vals = []
        dist_vals = [] 

        model.train()
        # Initialise training loop
        for idx, (images, labels, lesion_mask, action_identifier, tumour_centroid) in enumerate(train_dataloader):
            
        #print(f'\n Idx train : {idx}')
            lesion_projection = np.max(lesion_mask.detach().numpy(), axis = 3)

            # Move to GPU 
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()

            # Training steps 
            optimiser.zero_grad()
            pred_actions, _, _ = model(images)  # Obtain predicted segmentation masks 

            # Clamp actions between -1,1 as done with labelled examples 
            if clip_actions:
                pred_actions = torch.clamp(pred_actions, -1, 1)
            
            loss = loss_fn(pred_actions.float(), labels.float()) # Compute the masks 
            loss.backward() # Backward propagation of gradients with respect to loss 
            optimiser.step() 

            # Debugging print functions
            #print(f'Unique vals : {np.unique(images.detach().cpu())}')
            #print(f'Unique vals : {np.unique(pred_masks.detach().cpu())}')
            #print(f'Unique vals label: {np.unique(labels.detach().cpu())}')
            #print(f'Loss : {loss.item()}')

            # Compute metrics for each mini batch and append to statistics for each epoch
            with torch.no_grad():
                accuracy = compute_rmse(loss)
            #    accuracy, precision, recall  = compute_accuracy(pred_actions[:,:,0:13, 0:13, :], labels)
            #    print(f'RMSE : {accuracy.item()}')
                acc_vals.append(accuracy)

                hit_rate, ccl = compute_HR_CCL(images, pred_actions, lesion_projection, lesion_mask)
                dist, mean_dist  = compute_dist_to_lesion(images, pred_actions, action_identifier, tumour_centroid, lesion_mask)
                
                # Only add HR and CCL for fired positions; not for non-fired 
                if np.all(hit_rate != None):
                    hr_vals.append(hit_rate)
                if np.all(ccl != None):
                    ccl_vals.append(ccl)
                
                dist_vals.append(dist)

            loss_vals.append(loss.item())

            # Print loss every nth minibatch and dice score 
            #if idx % freq_print == 0: 
            #    print(f'Epoch {epoch_no} minibatch {idx} : loss : {loss.item():05f}, acc score : {acc.item():05f}')
            
        # Obtain mean dice loss and acc over this epoch, save to tensorboard
        with torch.no_grad():
            acc_epoch = torch.mean(torch.tensor((acc_vals)))
            loss_epoch = torch.mean(torch.tensor((loss_vals)))
            hr_epoch = torch.mean(torch.tensor(np.concatenate(hr_vals)))
            ccl_epoch = torch.mean(torch.tensor(np.concatenate(ccl_vals)))
            dist_epoch = torch.mean(torch.tensor(np.concatenate(dist_vals)))

        print(f'\n Epoch : {epoch_no} Average loss : {loss_epoch:5f} average RMSE {acc_epoch:5f}')
        print(f'Metrics : HR {hr_epoch}, CCL : {ccl_epoch}, dist metric : {dist_epoch}')

        with open(csv_train_path, 'a') as fp: 
            loss_points = np.stack([epoch_no, loss_epoch]).reshape(1,-1)
            np.savetxt(fp, loss_points, '%s', delimiter =",")

        # Save for all_loss_train
        all_loss_train[epoch_no] = loss_epoch
        all_acc_train[epoch_no] = acc_epoch 
        
        #Tensorboard saving 
        writer.add_scalar('Loss/train', loss_epoch, epoch_no)
        writer.add_scalar('RMSE/train', acc_epoch, epoch_no)
        writer.add_scalar('HR/train', hr_epoch, epoch_no)
        writer.add_scalar('CCL/train', ccl_epoch, epoch_no)
        writer.add_scalar('dist/train', dist_epoch, epoch_no)
    
        # Save newest model 
        train_model_path = os.path.join(train_folder, 'train_model.pth')
        torch.save(model.state_dict(), train_model_path)

        # Validate every nth epoch and save every nth mini batch 
        
        if epoch_no % freq_eval == 0: 
            print("VALIDATION : \n ")
            model.eval()        
            mean_loss, mean_acc, mean_hr, mean_ccl, mean_dist = validate_pertimestep(val_dataloader, model, use_cuda = use_cuda, save_path = train_folder, save_images = False, clip_actions = clip_actions)
            print(f'Validation loss for epoch {epoch_no} Average loss : {mean_loss:5f} average acc {mean_acc:5f}')
            print(f'Validation metrics : Mean HR {mean_hr} mean ccl : {mean_ccl}, mean dist : {mean_dist}')
            all_loss_val.append(mean_loss)
            all_acc_val.append(mean_acc)
            
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, mean_loss, mean_acc]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

            #Tensorboard saving
            writer.add_scalar('Loss/val', mean_loss, epoch_no)
            writer.add_scalar('RMSE/val', mean_acc, epoch_no)
            writer.add_scalar('mean_HR/val', mean_hr, epoch_no)
            writer.add_scalar('mean_ccl/val', mean_ccl, epoch_no)
            writer.add_scalar('mean_dist/val', mean_dist, epoch_no)

            if mean_loss < best_loss: 
                
                # Save best model as best validation model 
                val_model_path = os.path.join(train_folder, 'best_val_model.pth')
                torch.save(model.state_dict(), val_model_path)
        
                # Use as new best loss
                best_loss = mean_loss 

                # Save agent
                agent.policy = model 
                agent.save(os.path.join(save_folder, "agent_model"))
        
        elif epoch_no == 0: 
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, 1]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")
        else:
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, all_loss_val[-1]]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

        #print('Chicken') 
    return all_loss_train, all_loss_val, all_acc_train, all_acc_val 

def validate_pertimestep(val_dataloader, model, use_cuda = True, save_path = 'model_1', save_images = False, metric = 'rmse', clip_actions = False):

    # Set to evaluation mode 
    model.eval()
    acc_vals_eval = [] 
    loss_vals_eval = [] 
    hr_vals_eval = [] 
    ccl_vals_eval = [] 
    dist_vals_eval = [] 

    loss_fn = torch.nn.BCEWithLogitsLoss()
    for idx, (images, labels, lesion_mask, action_identifier, tumour_centroid) in enumerate(val_dataloader):
        
        lesion_projection = np.max(lesion_mask.detach().numpy(), axis = 3)

        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            output, _, _ = model(images)

            # Clamp actions between -1,1 as done with labelled examples 
            if clip_actions:
                output = torch.clamp(output, -1, 1)
            print(f"pred action : {output}")
                
            loss = loss_fn(output.float(), labels.float()) 
            acc = compute_rmse(loss)

            loss_vals_eval.append(loss.item())
            acc_vals_eval.append(acc)

            hit_rate, ccl = compute_HR_CCL(images, labels, lesion_projection, lesion_mask)
            dist, mean_dist = compute_dist_to_lesion(images, labels, action_identifier, tumour_centroid, lesion_mask)

            if np.all(hit_rate != None):
                hr_vals_eval.append(hit_rate)
            if np.all(hit_rate != None):
                ccl_vals_eval.append(ccl)

            dist_vals_eval.append(dist)


        #if save_images:
        #    # Save image, labels and outputs into h5py files
        #    img_name = patient_name[0].split(".")[0] + '_rectum_PRED.nrrd'
        #    img_path = os.path.join(save_path, img_name)
        #    sitk.WriteImage(sitk.GetImageFromArray(images.cpu()), img_path)
    
    with torch.no_grad():
        mean_acc = torch.mean(torch.FloatTensor(acc_vals_eval))
        mean_loss = torch.mean(torch.FloatTensor(loss_vals_eval))
        mean_hr = torch.mean(torch.FloatTensor(np.concatenate(hr_vals_eval)))
        mean_ccl = torch.mean(torch.FloatTensor(np.concatenate(ccl_vals_eval)))
        mean_dist = torch.mean(torch.FloatTensor(np.concatenate(dist_vals_eval)))

    return mean_loss, mean_acc, mean_hr, mean_ccl, mean_dist 

### Train scripts using new observations : 5 x 100 x 100 x 24 where 5 is num of channels 
def train_pertimestep_new(model, agent, train_dataloader, val_dataloader, num_epochs = 10, use_cuda = False, save_folder = 'model_1', loss_fn_str = 'MSE', clip_actions = False):
    
    """
    A function that performs the training and validation loop 
    :params:
    :prostate_dataloader: torch dataloader that can load images and masks in
    :num_epochs: (int) Number of epochs to train for 
    """
    current_dir = os.getcwd()
    train_folder = os.path.join(current_dir, save_folder)

    os.makedirs(train_folder, exist_ok = True) 
    print(f'train folder path : {train_folder}')
    writer = SummaryWriter(os.path.join(train_folder, 'runs')) 

    if use_cuda:
        model.cuda()
    
    # Defining optimiser, loss fn 
    #loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.MSELoss()
    print('loss fn : MSE Loss')

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-05)

    # Parameters 
    step = 0 
    freq_print = 4
    freq_eval = 4

    # Saving files 
    all_loss_train = np.zeros((num_epochs,1))
    all_acc_train = np.zeros((num_epochs, 1))
    all_loss_val = [] 
    all_acc_val = []
    best_loss = np.inf 
    best_acc = 0 
    
    csv_train_path = os.path.join(train_folder, 'train_loss.csv')
    csv_val_path = os.path.join(train_folder, 'val_loss.csv')

    with open(csv_train_path, 'w') as fp:
        fp.write('''epoch_num, bce_loss''')
        fp.write('\n')

    with open(csv_val_path, 'w') as fp:
        fp.write('''epoch_num, bce_loss''')
        fp.write('\n')

    
    for epoch_no in range(num_epochs):
        
        print(f"\n Epoch number : {epoch_no}")
        acc_vals = []
        loss_vals = [] 
        hr_vals = [] 
        ccl_vals = []
        dist_vals = [] 
        mse_x = [] 
        mse_y = []
        mse_z = []

        model.train()
        # Initialise training loop
        for idx, (images, labels, lesion_mask, action_identifier, tumour_centroid, grid_pos, max_depth) in enumerate(train_dataloader):
            
        #print(f'\n Idx train : {idx}')
            lesion_projection = np.max(lesion_mask.detach().numpy(), axis = 3)

            # Move to GPU 
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()

            # Training steps 
            optimiser.zero_grad()
            pred_actions, _, _ = model(images)  # Obtain predicted segmentation masks 

            # Clamp actions between -1,1 as done with labelled examples 
            if clip_actions:
                pred_actions = torch.clamp(pred_actions, -1, 1)
            
            loss = loss_fn(pred_actions.float(), labels.float()) # Compute the masks 
            loss.backward() # Backward propagation of gradients with respect to loss 
            optimiser.step() 

            # compute mse loss per action
            mse_actions = compute_mse_actions(pred_actions.float(), labels.float())
            mse_x.append(mse_actions[0].cpu())
            mse_y.append(mse_actions[1].cpu())
            mse_z.append(mse_actions[2].cpu())   
            
            # Debugging print functions
            #print(f'Unique vals : {np.unique(images.detach().cpu())}')
            #print(f'Unique vals : {np.unique(pred_masks.detach().cpu())}')
            #print(f'Unique vals label: {np.unique(labels.detach().cpu())}')
            #print(f'Loss : {loss.item()}')

            # Compute metrics for each mini batch and append to statistics for each epoch
            with torch.no_grad():
                accuracy = compute_rmse(loss)
            #    accuracy, precision, recall  = compute_accuracy(pred_actions[:,:,0:13, 0:13, :], labels)
            #    print(f'RMSE : {accuracy.item()}')
                acc_vals.append(accuracy)

                hit_rate, ccl = compute_HR(images, pred_actions, grid_pos, max_depth) #, lesion_projection, lesion_mask)
                #dist, mean_dist  = compute_dist_to_lesion(images, pred_actions, action_identifier, tumour_centroid, lesion_mask)
                dist, mean_dist = compute_DIST(images, pred_actions, tumour_centroid)
                
                # Only add HR and CCL for fired positions; not for non-fired 
                if np.all(hit_rate != None):
                    hr_vals.append(hit_rate)
                if np.all(ccl != None):
                    ccl_vals.append(ccl)
                
                dist_vals.append(dist)

            loss_vals.append(loss.item())

            # Print loss every nth minibatch and dice score 
            #if idx % freq_print == 0: 
            #    print(f'Epoch {epoch_no} minibatch {idx} : loss : {loss.item():05f}, acc score : {acc.item():05f}')
            
        # Obtain mean dice loss and acc over this epoch, save to tensorboard
        with torch.no_grad():
            acc_epoch = torch.mean(torch.tensor((acc_vals)))
            loss_epoch = torch.mean(torch.tensor((loss_vals)))

            # metrics 
            hr_epoch = torch.mean(torch.tensor(np.concatenate(hr_vals)).float())
            ccl_epoch = torch.mean(torch.tensor(np.concatenate(ccl_vals)).float())
            dist_epoch = torch.mean(torch.tensor(np.concatenate(dist_vals)).float())
            mse_x_epoch = torch.mean(torch.tensor(np.stack(mse_x)).float())
            mse_y_epoch = torch.mean(torch.tensor(np.stack(mse_y)).float())    
            mse_z_epoch = torch.mean(torch.tensor(np.stack(mse_z)).float())
            
            # compute std 
            hr_std = torch.mean(torch.tensor(np.concatenate(hr_vals)).float())
            ccl_std = torch.mean(torch.tensor(np.concatenate(ccl_vals)).float())
            dist_std = torch.mean(torch.tensor(np.concatenate(dist_vals)).float())


        print(f'\n Epoch : {epoch_no} Average loss : {loss_epoch:5f} average RMSE {acc_epoch:5f}')
        print(f'Metrics : HR {hr_epoch} +- {hr_std}, CCL : {ccl_epoch} +- {ccl_std}, dist metric : {dist_epoch} +- {dist_std}')
        print(f'Metrics : mse_x {mse_x_epoch} mse_y {mse_y_epoch} mse_z {mse_z_epoch}')

        with open(csv_train_path, 'a') as fp: 
            loss_points = np.stack([epoch_no, loss_epoch]).reshape(1,-1)
            np.savetxt(fp, loss_points, '%s', delimiter =",")

        # Save for all_loss_train
        all_loss_train[epoch_no] = loss_epoch
        all_acc_train[epoch_no] = acc_epoch 
        
        #Tensorboard saving 
        writer.add_scalar('Loss/train', loss_epoch, epoch_no)
        writer.add_scalar('RMSE/train', acc_epoch, epoch_no)
        writer.add_scalar('HR/train', hr_epoch, epoch_no)
        writer.add_scalar('CCL/train', ccl_epoch, epoch_no)
        writer.add_scalar('dist/train', dist_epoch, epoch_no)
        writer.add_scalar('mse_x/train', mse_x_epoch, epoch_no)
        writer.add_scalar('mse_y/train', mse_y_epoch, epoch_no)
        writer.add_scalar('mse_z/train', mse_z_epoch, epoch_no)
    
        # Save newest model 
        train_model_path = os.path.join(train_folder, 'train_model.pth')
        torch.save(model.state_dict(), train_model_path)

        # Validate every nth epoch and save every nth mini batch 
        
        if epoch_no % freq_eval == 0: 
            print("VALIDATION : \n ")
            model.eval()        
            mean_loss, mean_acc, mean_hr, mean_ccl, mean_dist, mean_x, mean_y, mean_z = validate_pertimestep_new(val_dataloader, model, use_cuda = use_cuda, save_path = train_folder, save_images = False, clip_actions = clip_actions)
            print(f'Validation loss for epoch {epoch_no} Average loss : {mean_loss:5f} average acc {mean_acc:5f}')
            all_loss_val.append(mean_loss)
            all_acc_val.append(mean_acc)
            
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, mean_loss, mean_acc]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

            #Tensorboard saving
            writer.add_scalar('Loss/val', mean_loss, epoch_no)
            writer.add_scalar('RMSE/val', mean_acc, epoch_no)
            writer.add_scalar('mean_HR/val', mean_hr, epoch_no)
            writer.add_scalar('mean_ccl/val', mean_ccl, epoch_no)
            writer.add_scalar('mean_dist/val', mean_dist, epoch_no)
            writer.add_scalar('mse_x/val', mean_x, epoch_no)
            writer.add_scalar('mse_y/val', mean_y, epoch_no)
            writer.add_scalar('mse_z/val', mean_z, epoch_no)

            if mean_loss < best_loss: 
                
                # Save best model as best validation model 
                val_model_path = os.path.join(train_folder, 'best_val_model.pth')
                torch.save(model.state_dict(), val_model_path)
        
                # Use as new best loss
                best_loss = mean_loss 

                # Save agent
                agent.policy = model 
                agent.save(os.path.join(save_folder, "agent_model"))
        
        elif epoch_no == 0: 
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, 1]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")
        else:
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, all_loss_val[-1]]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

        #print('Chicken') 
    return all_loss_train, all_loss_val, all_acc_train, all_acc_val 

class HR_loss_fn():
    """
    A loss function that combines both HR and MSE as loss functions
    """
    
    def __init__(self, weight = 0.5):
        self.mse_loss = torch.nn.MSELoss()
        self.weight = weight
    
    def __call__(self, pred, label, hr):
        """
        Computes a weighted sum of the mse loss and the HR loss

        Parameters:
        ------------
        :pred: Predicted actions
        :label: Ground truth actions
        :hr: computed HR when executing predicted actions on env
        """
        
        mse_loss = self.mse_loss(pred.float(), label.float())

        # Combination loss of MSE vs HR
        combined_loss = self.weight*mse_loss - (1-self.weight)*hr
        
        return combined_loss
        
def train_pertimestep_GS_HR(model, agent, train_dataloader, val_dataloader, num_epochs = 10, use_cuda = False, save_folder = 'model_1', loss_fn_str = 'MSE', clip_actions = False, device = 'cpu'):
    
    """
    A function that performs the training and validation loop 
    :params:
    :prostate_dataloader: torch dataloader that can load images and masks in
    :num_epochs: (int) Number of epochs to train for 
    """
    current_dir = os.getcwd()
    train_folder = os.path.join(current_dir, save_folder)

    os.makedirs(train_folder, exist_ok = True) 
    print(f'train folder path : {train_folder}')
    writer = SummaryWriter(os.path.join(train_folder, 'runs')) 
    
    if use_cuda:
        print(f'Using cuda')
        model.cuda()
    
    # Defining optimiser, loss fn 
    #loss_fn = torch.nn.BCELoss()
    loss_fn = HR_loss_fn()
    print('loss fn : HR Loss')

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-05)

    # Parameters 
    step = 0 
    freq_print = 4
    freq_eval = 4

    # Saving files 
    all_loss_train = np.zeros((num_epochs,1))
    all_acc_train = np.zeros((num_epochs, 1))
    all_loss_val = [] 
    all_acc_val = []
    best_loss = np.inf 
    best_acc = 0 
    
    csv_train_path = os.path.join(train_folder, 'train_loss.csv')
    csv_val_path = os.path.join(train_folder, 'val_loss.csv')

    with open(csv_train_path, 'w') as fp:
        fp.write('''epoch_num, bce_loss''')
        fp.write('\n')

    with open(csv_val_path, 'w') as fp:
        fp.write('''epoch_num, bce_loss''')
        fp.write('\n')

    # Initialise biopsy env to interact with 
    biopsy_env = SimBiopsyEnv(device)
    
    for epoch_no in range(num_epochs):
        
        print(f"\n Epoch number : {epoch_no}")
        acc_vals = []
        loss_vals = [] 
        hr_vals = [] 
        ccl_vals = []
        dist_vals = [] 
        mse_x = [] 
        mse_y = []
        mse_z = []
        
        # Include ground truth hr, ccl vals
        hr_vals_gt = []
        ccl_vals_gt = [] 

        model.train()
        # Initialise training loop
        for idx, (images, labels, prostate_centroid, tumour_centroid, lesion_mask, max_depth) in enumerate(train_dataloader):

        #print(f'\n Idx train : {idx}')
            lesion_projection = np.max(lesion_mask.detach().numpy(), axis = 3)

            # Move to GPU 
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()

            # Training steps 
            optimiser.zero_grad()
            pred_actions, _, _ = model(images)  # Obtain predicted segmentation masks 

            # Clamp actions between -1,1 as done with labelled examples 
            if clip_actions:
                pred_actions = torch.clamp(pred_actions, -1, 1)
            
            # Compute HR from biopsy env 
            grid_pos = compute_grid_pos(images)
            grid_pos = grid_pos.to(device)
            
            # Compute hr, ccl from predicted actions 
            new_pos = biopsy_env.find_new_needle_pos(pred_actions, grid_pos)
            new_obs = biopsy_env.obtain_new_obs(new_pos, max_depth)
            hr, ccl = biopsy_env.compute_HR(images[:,0,:,:,:], new_obs, pred_actions)
            
            mean_hr_loss = torch.mean(hr).float()
            loss = loss_fn(pred_actions.float(), labels.float(), mean_hr_loss) # Compute the masks 
            loss.backward() # Backward propagation of gradients with respect to loss 
            optimiser.step() 

            # compute mse loss per action
            mse_actions = compute_mse_actions(pred_actions.float(), labels.float())
            mse_x.append(mse_actions[0].cpu())
            mse_y.append(mse_actions[1].cpu())
            mse_z.append(mse_actions[2].cpu())   
            
            # Debugging print functions
            #print(f'Unique vals : {np.unique(images.detach().cpu())}')
            #print(f'Unique vals : {np.unique(pred_masks.detach().cpu())}')
            #print(f'Unique vals label: {np.unique(labels.detach().cpu())}')
            #print(f'Loss : {loss.item()}')

            # Compute metrics for each mini batch and append to statistics for each epoch
            with torch.no_grad():
                accuracy = compute_rmse(loss)
            #    accuracy, precision, recall  = compute_accuracy(pred_actions[:,:,0:13, 0:13, :], labels)
            #    print(f'RMSE : {accuracy.item()}')
                acc_vals.append(accuracy)

                grid_pos = compute_grid_pos(images)
                grid_pos = grid_pos.to(device)
                
                # Compute hr, ccl from predicted actions 
                new_pos = biopsy_env.find_new_needle_pos(pred_actions, grid_pos)
                new_obs = biopsy_env.obtain_new_obs(new_pos, max_depth)
                hr, ccl = biopsy_env.compute_HR(images[:,0,:,:,:], new_obs, pred_actions)
                dist, mean_dist = compute_DIST(images, pred_actions, tumour_centroid)
                
                # Compute gt values 
                new_pos_gt = biopsy_env.find_new_needle_pos(labels, grid_pos)
                new_obs_gt = biopsy_env.obtain_new_obs(new_pos_gt, max_depth)
                hr_gt, ccl_gt = biopsy_env.compute_HR(images[:,0,:,:,:], new_obs_gt, labels)
                
                print(f"New pos (pred) : {new_pos[0]} | (gt) : {new_pos_gt[0]}")
                print(f"HR (pred vs gt) : {torch.mean(hr)} | {torch.mean(hr_gt)} \n CCL (pred vs gt) : {torch.mean(ccl)} | {torch.mean(ccl_gt)}")
                
                # # groudn truth actions
                # new_pos = biopsy_env.find_new_needle_pos(labels, grid_pos)
                # new_obs = biopsy_env.obtain_new_obs(new_pos, max_depth)
                # hr, ccl = biopsy_env.compute_HR(images[:,0,:,:,:], new_obs, labels)
                
                #TODO : fix compute HR / CCL FUNCTION FOR TRAINING!!! 
                #hit_rate, ccl = compute_HR(images, labels, grid_pos, max_depth) #, lesion_projection, lesion_mask)
                #dist, mean_dist  = compute_dist_to_lesion(images, pred_actions, action_identifier, tumour_centroid, lesion_mask)
                #dist, mean_dist = compute_DIST(images, pred_actions, tumour_centroid)
                
                # Only add HR and CCL for fired positions; not for non-fired 
                if np.all(hr != None):
                    hr_vals.append(hr)
                if np.all(ccl != None):
                    ccl_vals.append(ccl)
                    
                if np.all(hr_gt != None):
                    hr_vals_gt.append(hr_gt)
                if np.all(ccl_gt != None):
                    ccl_vals_gt.append(ccl_gt)
                    
                
                dist_vals.append(dist)

            loss_vals.append(loss.item())

            # Print loss every nth minibatch and dice score 
            #if idx % freq_print == 0: 
            #    print(f'Epoch {epoch_no} minibatch {idx} : loss : {loss.item():05f}, acc score : {acc.item():05f}')
            
        # Obtain mean dice loss and acc over this epoch, save to tensorboard
        with torch.no_grad():
            acc_epoch = torch.mean(torch.tensor((acc_vals)))
            loss_epoch = torch.mean(torch.tensor((loss_vals)))

            # metrics 
            hr_epoch = torch.mean(torch.FloatTensor(torch.cat(hr_vals)))
            ccl_epoch = torch.mean(torch.FloatTensor(torch.cat(ccl_vals)))
            dist_epoch = torch.mean(torch.tensor(np.concatenate(dist_vals)))
            mse_x_epoch = torch.mean(torch.tensor(np.stack(mse_x)).float())
            mse_y_epoch = torch.mean(torch.tensor(np.stack(mse_y)).float())    
            mse_z_epoch = torch.mean(torch.tensor(np.stack(mse_z)).float())
            
            # for ground truth labels
            hr_epoch_gt = torch.mean(torch.FloatTensor(torch.cat(hr_vals_gt)))
            ccl_epoch_gt = torch.mean(torch.FloatTensor(torch.cat(ccl_vals_gt)))
            
            # compute std 
            hr_std = torch.mean(torch.tensor(np.concatenate(hr_vals)).float())
            ccl_std = torch.mean(torch.tensor(np.concatenate(ccl_vals)).float())
            dist_std = torch.mean(torch.tensor(np.concatenate(dist_vals)).float())
            
            hr_std_gt = torch.std(torch.FloatTensor(torch.cat(hr_vals_gt)))
            ccl_std_gt = torch.std(torch.FloatTensor(torch.cat(ccl_vals_gt)))
            
        print(f'\n Epoch : {epoch_no} Average loss : {loss_epoch:5f} average RMSE {acc_epoch:5f}')
        print(f'Metrics : HR {hr_epoch} +- {hr_std}, CCL : {ccl_epoch} +- {ccl_std}, dist metric : {dist_epoch} +- {dist_std}')
        print(f'Metrics : mse_x {mse_x_epoch} mse_y {mse_y_epoch} mse_z {mse_z_epoch}')
        
        print(f'\n Ground truth : HR {hr_epoch_gt} +- {hr_std_gt}, CCL : {ccl_epoch_gt} +- {ccl_std_gt}')

        with open(csv_train_path, 'a') as fp: 
            loss_points = np.stack([epoch_no, loss_epoch]).reshape(1,-1)
            np.savetxt(fp, loss_points, '%s', delimiter =",")

        # Save for all_loss_train
        all_loss_train[epoch_no] = loss_epoch
        all_acc_train[epoch_no] = acc_epoch 
        
        #Tensorboard saving 
        writer.add_scalar('Loss/train', loss_epoch, epoch_no)
        writer.add_scalar('RMSE/train', acc_epoch, epoch_no)
        writer.add_scalar('HR/train', hr_epoch, epoch_no)
        writer.add_scalar('CCL/train', ccl_epoch, epoch_no)
        writer.add_scalar('dist/train', dist_epoch, epoch_no)
        writer.add_scalar('mse_x/train', mse_x_epoch, epoch_no)
        writer.add_scalar('mse_y/train', mse_y_epoch, epoch_no)
        writer.add_scalar('mse_z/train', mse_z_epoch, epoch_no)
        
        # Add for GT also 
        writer.add_scalar('HR/train_gt', hr_epoch_gt, epoch_no)
        writer.add_scalar('CCL/train_gt', ccl_epoch_gt, epoch_no)
    
        # Save newest model 
        train_model_path = os.path.join(train_folder, 'train_model.pth')
        torch.save(model.state_dict(), train_model_path)

        # Validate every nth epoch and save every nth mini batch 
        
        if epoch_no % freq_eval == 0: 
            print("VALIDATION : \n ")
            model.eval()        
            mean_loss, mean_acc, mean_hr, mean_ccl, mean_dist, mean_x, mean_y, mean_z = validate_pertimestep_GS_HR(val_dataloader, model, use_cuda = use_cuda, save_path = train_folder, save_images = False, clip_actions = clip_actions, device = device)
            print(f'Validation loss for epoch {epoch_no} Average loss : {mean_loss:5f} average acc {mean_acc:5f}')
            all_loss_val.append(mean_loss)
            all_acc_val.append(mean_acc)
            
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, mean_loss, mean_acc]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

            #Tensorboard saving
            writer.add_scalar('Loss/val', mean_loss, epoch_no)
            writer.add_scalar('RMSE/val', mean_acc, epoch_no)
            writer.add_scalar('HR/val', mean_hr, epoch_no)
            writer.add_scalar('CCL/val', mean_ccl, epoch_no)
            writer.add_scalar('dist/val', mean_dist, epoch_no)
            writer.add_scalar('mse_x/val', mean_x, epoch_no)
            writer.add_scalar('mse_y/val', mean_y, epoch_no)
            writer.add_scalar('mse_z/val', mean_z, epoch_no)

            if mean_loss < best_loss: 
                
                # Save best model as best validation model 
                val_model_path = os.path.join(train_folder, 'best_val_model.pth')
                torch.save(model.state_dict(), val_model_path)
        
                # Use as new best loss
                best_loss = mean_loss 

                # Save agent
                agent.policy = model 
                agent.save(os.path.join(save_folder, "agent_model"))
        
        elif epoch_no == 0: 
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, 1]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")
        else:
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, all_loss_val[-1]]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

        #print('Chicken') 
    return all_loss_train, all_loss_val, all_acc_train, all_acc_val 

def validate_pertimestep_GS_HR(val_dataloader, model, use_cuda = True, save_path = 'model_1', save_images = False, metric = 'rmse', clip_actions = False, device = 'cpu'):

    # Set to evaluation mode 
    model.eval()
    acc_vals_eval = [] 
    loss_vals_eval = [] 
    hr_vals_eval = [] 
    ccl_vals_eval = [] 
    dist_vals_eval = [] 
    mse_x = []
    mse_y = []
    mse_z = []
    
    biopsy_env = SimBiopsyEnv(device)

    #loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = HR_loss_fn()
    
    for idx, (images, labels, prostate_centroid, tumour_centroid, lesion_mask, max_depth) in enumerate(val_dataloader):
        
        lesion_projection = np.max(lesion_mask.detach().numpy(), axis = 3)

        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            output, _, _ = model(images)

            # Clamp actions between -1,1 as done with labelled examples 
            if clip_actions:
                output = torch.clamp(output, -1, 1)
            print(f"pred action : {output}")
                
        
            mse_actions = compute_mse_actions(output.float(), labels.float())
            mse_x.append(mse_actions[0].cpu())
            mse_y.append(mse_actions[1].cpu())
            mse_z.append(mse_actions[2].cpu())   
        
            grid_pos = compute_grid_pos(images)
            grid_pos = grid_pos.to(device)
            new_pos = biopsy_env.find_new_needle_pos(output, grid_pos)
            new_obs = biopsy_env.obtain_new_obs(new_pos, max_depth)
            hr, ccl = biopsy_env.compute_HR(images[:,0,:,:,:], new_obs, output)
            dist, mean_dist = compute_DIST(images, output, tumour_centroid)
            
            # Compute loss using hr loss function! 
            mean_hr_sub = torch.mean(hr).float()
            loss = loss_fn(output.float(), labels.float(), mean_hr_sub) 
            acc = compute_rmse(loss)
            loss_vals_eval.append(loss.item())
            acc_vals_eval.append(acc)
            
            #hit_rate, ccl = compute_HR(images, labels, grid_pos, max_depth) #, lesion_projection, lesion_mask)
            #dist, mean_dist  = compute_dist_to_lesion(images, pred_actions, action_identifier, tumour_centroid, lesion_mask)
            #dist, mean_dist = compute_DIST(images, labels, tumour_centroid)


            if np.all(hr != None):
                hr_vals_eval.append(hr)
            if np.all(ccl != None):
                ccl_vals_eval.append(ccl)

            dist_vals_eval.append(dist)


        #if save_images:
        #    # Save image, labels and outputs into h5py files
        #    img_name = patient_name[0].split(".")[0] + '_rectum_PRED.nrrd'
        #    img_path = os.path.join(save_path, img_name)
        #    sitk.WriteImage(sitk.GetImageFromArray(images.cpu()), img_path)
    
    with torch.no_grad():
        mean_acc = torch.mean(torch.FloatTensor(acc_vals_eval))
        mean_loss = torch.mean(torch.FloatTensor(loss_vals_eval))
        mean_hr = torch.mean(torch.FloatTensor((torch.cat(hr_vals_eval))))
        mean_ccl = torch.mean(torch.FloatTensor(torch.cat((ccl_vals_eval))))
        mean_dist = torch.mean(torch.FloatTensor(np.concatenate(dist_vals_eval)))
        mse_x_epoch = torch.mean(torch.FloatTensor(np.stack(mse_x)))
        mse_y_epoch = torch.mean(torch.FloatTensor(np.stack(mse_y)))
        mse_z_epoch = torch.mean(torch.FloatTensor(np.stack(mse_z)))

    return mean_loss, mean_acc, mean_hr, mean_ccl, mean_dist, mse_x_epoch, mse_y_epoch, mse_z_epoch


def train_pertimestep_GS(model, agent, train_dataloader, val_dataloader, num_epochs = 10, use_cuda = False, save_folder = 'model_1', loss_fn_str = 'MSE', clip_actions = False, device = 'cpu'):
    
    """
    A function that performs the training and validation loop 
    :params:
    :prostate_dataloader: torch dataloader that can load images and masks in
    :num_epochs: (int) Number of epochs to train for 
    """
    current_dir = os.getcwd()
    train_folder = os.path.join(current_dir, save_folder)

    os.makedirs(train_folder, exist_ok = True) 
    print(f'train folder path : {train_folder}')
    writer = SummaryWriter(os.path.join(train_folder, 'runs')) 
    
    if use_cuda:
        print(f'Using cuda')
        model.cuda()
    
    # Defining optimiser, loss fn 
    #loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.MSELoss()
    print('loss fn : MSE Loss')

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-05)

    # Parameters 
    step = 0 
    freq_print = 4
    freq_eval = 4

    # Saving files 
    all_loss_train = np.zeros((num_epochs, 1))
    all_acc_train = np.zeros((num_epochs, 1))
    all_loss_val = [] 
    all_acc_val = []
    best_loss = np.inf 
    best_acc = 0 
    
    csv_train_path = os.path.join(train_folder, 'train_loss.csv')
    csv_val_path = os.path.join(train_folder, 'val_loss.csv')

    with open(csv_train_path, 'w') as fp:
        fp.write('''epoch_num, bce_loss''')
        fp.write('\n')

    with open(csv_val_path, 'w') as fp:
        fp.write('''epoch_num, bce_loss''')
        fp.write('\n')

    # Initialise biopsy env to interact with 
    biopsy_env = SimBiopsyEnv(device)
    
    for epoch_no in range(num_epochs):
        
        print(f"\n Epoch number : {epoch_no}")
        acc_vals = []
        loss_vals = [] 
        hr_vals = [] 
        ccl_vals = []
        dist_vals = [] 
        mse_x = [] 
        mse_y = []
        mse_z = []
        
        # Include ground truth hr, ccl vals
        hr_vals_gt = []
        ccl_vals_gt = [] 

        model.train()
        # Initialise training loop
        for idx, (images, labels, prostate_centroid, tumour_centroid, lesion_mask, max_depth) in enumerate(train_dataloader):

        #print(f'\n Idx train : {idx}')
            lesion_projection = np.max(lesion_mask.detach().numpy(), axis = 3)

            # Move to GPU 
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()

            # Training steps 
            optimiser.zero_grad()
            pred_actions, _, _ = model(images)  # Obtain predicted segmentation masks 

            # Clamp actions between -1,1 as done with labelled examples 
            if clip_actions:
                pred_actions = torch.clamp(pred_actions, -1, 1)
            
            loss = loss_fn(pred_actions.float(), labels.float()) # Compute the masks 
            loss.backward() # Backward propagation of gradients with respect to loss 
            optimiser.step() 

            # compute mse loss per action
            mse_actions = compute_mse_actions(pred_actions.float(), labels.float())
            mse_x.append(mse_actions[0].cpu())
            mse_y.append(mse_actions[1].cpu())
            mse_z.append(mse_actions[2].cpu())   
            
            # Debugging print functions
            #print(f'Unique vals : {np.unique(images.detach().cpu())}')
            #print(f'Unique vals : {np.unique(pred_masks.detach().cpu())}')
            #print(f'Unique vals label: {np.unique(labels.detach().cpu())}')
            #print(f'Loss : {loss.item()}')

            # Compute metrics for each mini batch and append to statistics for each epoch
            with torch.no_grad():
                accuracy = compute_rmse(loss)
            #    accuracy, precision, recall  = compute_accuracy(pred_actions[:,:,0:13, 0:13, :], labels)
            #    print(f'RMSE : {accuracy.item()}')
                acc_vals.append(accuracy)

                grid_pos = compute_grid_pos(images)
                grid_pos = grid_pos.to(device)
                
                # Compute hr, ccl from predicted actions 
                new_pos = biopsy_env.find_new_needle_pos(pred_actions, grid_pos)
                new_obs = biopsy_env.obtain_new_obs(new_pos, max_depth)
                hr, ccl = biopsy_env.compute_HR(images[:,0,:,:,:], new_obs, pred_actions)
                dist, mean_dist = compute_DIST(images, pred_actions, tumour_centroid)
                
                # Compute gt values 
                new_pos_gt = biopsy_env.find_new_needle_pos(labels, grid_pos)
                new_obs_gt = biopsy_env.obtain_new_obs(new_pos_gt, max_depth)
                hr_gt, ccl_gt = biopsy_env.compute_HR(images[:,0,:,:,:], new_obs_gt, labels)
                
                print(f"New pos (pred) : {new_pos[0]} | (gt) : {new_pos_gt[0]}")
                print(f"HR (pred vs gt) : {torch.mean(hr)} | {torch.mean(hr_gt)} \n CCL (pred vs gt) : {torch.mean(ccl)} | {torch.mean(ccl_gt)}")
                
                # # groudn truth actions
                # new_pos = biopsy_env.find_new_needle_pos(labels, grid_pos)
                # new_obs = biopsy_env.obtain_new_obs(new_pos, max_depth)
                # hr, ccl = biopsy_env.compute_HR(images[:,0,:,:,:], new_obs, labels)
                
                #TODO : fix compute HR / CCL FUNCTION FOR TRAINING!!! 
                #hit_rate, ccl = compute_HR(images, labels, grid_pos, max_depth) #, lesion_projection, lesion_mask)
                #dist, mean_dist  = compute_dist_to_lesion(images, pred_actions, action_identifier, tumour_centroid, lesion_mask)
                #dist, mean_dist = compute_DIST(images, pred_actions, tumour_centroid)
                
                # Only add HR and CCL for fired positions; not for non-fired 
                if np.all(hr != None):
                    hr_vals.append(hr)
                if np.all(ccl != None):
                    ccl_vals.append(ccl)
                    
                if np.all(hr_gt != None):
                    hr_vals_gt.append(hr_gt)
                if np.all(ccl_gt != None):
                    ccl_vals_gt.append(ccl_gt)
                    
                
                dist_vals.append(dist)

            loss_vals.append(loss.item())

            # Print loss every nth minibatch and dice score 
            #if idx % freq_print == 0: 
            #    print(f'Epoch {epoch_no} minibatch {idx} : loss : {loss.item():05f}, acc score : {acc.item():05f}')
            
        # Obtain mean dice loss and acc over this epoch, save to tensorboard
        with torch.no_grad():
            acc_epoch = torch.mean(torch.tensor((acc_vals)))
            loss_epoch = torch.mean(torch.tensor((loss_vals)))

            # metrics 
            hr_epoch = torch.mean(torch.FloatTensor(torch.cat(hr_vals)))
            ccl_epoch = torch.mean(torch.FloatTensor(torch.cat(ccl_vals)))
            dist_epoch = torch.mean(torch.tensor(np.concatenate(dist_vals)))
            mse_x_epoch = torch.mean(torch.tensor(np.stack(mse_x)).float())
            mse_y_epoch = torch.mean(torch.tensor(np.stack(mse_y)).float())    
            mse_z_epoch = torch.mean(torch.tensor(np.stack(mse_z)).float())
            
            # for ground truth labels
            hr_epoch_gt = torch.mean(torch.FloatTensor(torch.cat(hr_vals_gt)))
            ccl_epoch_gt = torch.mean(torch.FloatTensor(torch.cat(ccl_vals_gt)))
            
            # compute std 
            hr_std = torch.mean(torch.tensor(np.concatenate(hr_vals)).float())
            ccl_std = torch.mean(torch.tensor(np.concatenate(ccl_vals)).float())
            dist_std = torch.mean(torch.tensor(np.concatenate(dist_vals)).float())
            
            hr_std_gt = torch.std(torch.FloatTensor(torch.cat(hr_vals_gt)))
            ccl_std_gt = torch.std(torch.FloatTensor(torch.cat(ccl_vals_gt)))
            
        print(f'\n Epoch : {epoch_no} Average loss : {loss_epoch:5f} average RMSE {acc_epoch:5f}')
        print(f'Metrics : HR {hr_epoch} +- {hr_std}, CCL : {ccl_epoch} +- {ccl_std}, dist metric : {dist_epoch} +- {dist_std}')
        print(f'Metrics : mse_x {mse_x_epoch} mse_y {mse_y_epoch} mse_z {mse_z_epoch}')
        
        print(f'\n Ground truth : HR {hr_epoch_gt} +- {hr_std_gt}, CCL : {ccl_epoch_gt} +- {ccl_std_gt}')

        with open(csv_train_path, 'a') as fp: 
            loss_points = np.stack([epoch_no, loss_epoch]).reshape(1,-1)
            np.savetxt(fp, loss_points, '%s', delimiter =",")

        # Save for all_loss_train
        all_loss_train[epoch_no] = loss_epoch
        all_acc_train[epoch_no] = acc_epoch 
        
        #Tensorboard saving 
        writer.add_scalar('Loss/train', loss_epoch, epoch_no)
        writer.add_scalar('RMSE/train', acc_epoch, epoch_no)
        writer.add_scalar('HR/train', hr_epoch, epoch_no)
        writer.add_scalar('CCL/train', ccl_epoch, epoch_no)
        writer.add_scalar('dist/train', dist_epoch, epoch_no)
        writer.add_scalar('mse_x/train', mse_x_epoch, epoch_no)
        writer.add_scalar('mse_y/train', mse_y_epoch, epoch_no)
        writer.add_scalar('mse_z/train', mse_z_epoch, epoch_no)
        
        # Add for GT also 
        writer.add_scalar('HR/train_gt', hr_epoch_gt, epoch_no)
        writer.add_scalar('CCL/train_gt', ccl_epoch_gt, epoch_no)
    
        # Save newest model 
        train_model_path = os.path.join(train_folder, 'train_model.pth')
        torch.save(model.state_dict(), train_model_path)

        # Validate every nth epoch and save every nth mini batch 
        
        if epoch_no % freq_eval == 0: 
            print("VALIDATION : \n ")
            model.eval()        
            mean_loss, mean_acc, mean_hr, mean_ccl, mean_dist, mean_x, mean_y, mean_z = validate_pertimestep_GS(val_dataloader, model, use_cuda = use_cuda, save_path = train_folder, save_images = False, clip_actions = clip_actions, device = device)
            print(f'Validation loss for epoch {epoch_no} Average loss : {mean_loss:5f} average acc {mean_acc:5f}')
            all_loss_val.append(mean_loss)
            all_acc_val.append(mean_acc)
            
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, mean_loss, mean_acc]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

            #Tensorboard saving
            writer.add_scalar('Loss/val', mean_loss, epoch_no)
            writer.add_scalar('RMSE/val', mean_acc, epoch_no)
            writer.add_scalar('HR/val', mean_hr, epoch_no)
            writer.add_scalar('CCL/val', mean_ccl, epoch_no)
            writer.add_scalar('dist/val', mean_dist, epoch_no)
            writer.add_scalar('mse_x/val', mean_x, epoch_no)
            writer.add_scalar('mse_y/val', mean_y, epoch_no)
            writer.add_scalar('mse_z/val', mean_z, epoch_no)

            if mean_loss < best_loss: 
                
                # Save best model as best validation model 
                val_model_path = os.path.join(train_folder, 'best_val_model.pth')
                torch.save(model.state_dict(), val_model_path)
        
                # Use as new best loss
                best_loss = mean_loss 

                # Save agent
                agent.policy = model 
                agent.save(os.path.join(save_folder, "agent_model"))
        
        elif epoch_no == 0: 
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, 1]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")
        else:
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, all_loss_val[-1]]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

        #print('Chicken') 
    return all_loss_train, all_loss_val, all_acc_train, all_acc_val 

def validate_pertimestep_new(val_dataloader, model, use_cuda = True, save_path = 'model_1', save_images = False, metric = 'rmse', clip_actions = False):

    # Set to evaluation mode 
    model.eval()
    acc_vals_eval = [] 
    loss_vals_eval = [] 
    hr_vals_eval = [] 
    ccl_vals_eval = [] 
    dist_vals_eval = [] 
    mse_x = []
    mse_y = []
    mse_z = []

    loss_fn = torch.nn.BCEWithLogitsLoss()
    for idx, (images, labels, lesion_mask, action_identifier, tumour_centroid, grid_pos, max_depth) in enumerate(val_dataloader):
        
        lesion_projection = np.max(lesion_mask.detach().numpy(), axis = 3)

        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            output, _, _ = model(images)

            # Clamp actions between -1,1 as done with labelled examples 
            if clip_actions:
                output = torch.clamp(output, -1, 1)
            print(f"pred action : {output}")
                
            loss = loss_fn(output.float(), labels.float()) 
            acc = compute_rmse(loss)
        
            mse_actions = compute_mse_actions(output.float(), labels.float())
            mse_x.append(mse_actions[0].cpu())
            mse_y.append(mse_actions[1].cpu())
            mse_z.append(mse_actions[2].cpu())   
            
            loss_vals_eval.append(loss.item())
            acc_vals_eval.append(acc)


            hit_rate, ccl = compute_HR(images, labels, grid_pos, max_depth) #, lesion_projection, lesion_mask)
            #dist, mean_dist  = compute_dist_to_lesion(images, pred_actions, action_identifier, tumour_centroid, lesion_mask)
            dist, mean_dist = compute_DIST(images, labels, tumour_centroid)
                

            if np.all(hit_rate != None):
                hr_vals_eval.append(hit_rate)
            if np.all(hit_rate != None):
                ccl_vals_eval.append(ccl)

            dist_vals_eval.append(dist)


        #if save_images:
        #    # Save image, labels and outputs into h5py files
        #    img_name = patient_name[0].split(".")[0] + '_rectum_PRED.nrrd'
        #    img_path = os.path.join(save_path, img_name)
        #    sitk.WriteImage(sitk.GetImageFromArray(images.cpu()), img_path)
    
    with torch.no_grad():
        mean_acc = torch.mean(torch.FloatTensor(acc_vals_eval))
        mean_loss = torch.mean(torch.FloatTensor(loss_vals_eval))
        mean_hr = torch.mean(torch.FloatTensor(np.concatenate(hr_vals_eval)))
        mean_ccl = torch.mean(torch.FloatTensor(np.concatenate(ccl_vals_eval)))
        mean_dist = torch.mean(torch.FloatTensor(np.concatenate(dist_vals_eval)))
        mse_x_epoch = torch.mean(torch.FloatTensor(np.stack(mse_x)))
        mse_y_epoch = torch.mean(torch.FloatTensor(np.stack(mse_y)))
        mse_z_epoch = torch.mean(torch.FloatTensor(np.stack(mse_z)))

    return mean_loss, mean_acc, mean_hr, mean_ccl, mean_dist, mse_x_epoch, mse_y_epoch, mse_z_epoch

def validate_pertimestep_GS(val_dataloader, model, use_cuda = True, save_path = 'model_1', save_images = False, metric = 'rmse', clip_actions = False, device = 'cpu'):

    # Set to evaluation mode 
    model.eval()
    acc_vals_eval = [] 
    loss_vals_eval = [] 
    hr_vals_eval = [] 
    ccl_vals_eval = [] 
    dist_vals_eval = [] 
    mse_x = []
    mse_y = []
    mse_z = []
    
    biopsy_env = SimBiopsyEnv(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    for idx, (images, labels, prostate_centroid, tumour_centroid, lesion_mask, max_depth) in enumerate(val_dataloader):
        
        lesion_projection = np.max(lesion_mask.detach().numpy(), axis = 3)

        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            output, _, _ = model(images)

            # Clamp actions between -1,1 as done with labelled examples 
            if clip_actions:
                output = torch.clamp(output, -1, 1)
            print(f"pred action : {output}")
                
            loss = loss_fn(output.float(), labels.float()) 
            acc = compute_rmse(loss)
        
            mse_actions = compute_mse_actions(output.float(), labels.float())
            mse_x.append(mse_actions[0].cpu())
            mse_y.append(mse_actions[1].cpu())
            mse_z.append(mse_actions[2].cpu())   
            
            loss_vals_eval.append(loss.item())
            acc_vals_eval.append(acc)

            grid_pos = compute_grid_pos(images)
            grid_pos = grid_pos.to(device)
            new_pos = biopsy_env.find_new_needle_pos(output, grid_pos)
            new_obs = biopsy_env.obtain_new_obs(new_pos, max_depth)
            hr, ccl = biopsy_env.compute_HR(images[:,0,:,:,:], new_obs, output)
            dist, mean_dist = compute_DIST(images, output, tumour_centroid)
            
            #hit_rate, ccl = compute_HR(images, labels, grid_pos, max_depth) #, lesion_projection, lesion_mask)
            #dist, mean_dist  = compute_dist_to_lesion(images, pred_actions, action_identifier, tumour_centroid, lesion_mask)
            #dist, mean_dist = compute_DIST(images, labels, tumour_centroid)


            if np.all(hr != None):
                hr_vals_eval.append(hr)
            if np.all(ccl != None):
                ccl_vals_eval.append(ccl)

            dist_vals_eval.append(dist)


        #if save_images:
        #    # Save image, labels and outputs into h5py files
        #    img_name = patient_name[0].split(".")[0] + '_rectum_PRED.nrrd'
        #    img_path = os.path.join(save_path, img_name)
        #    sitk.WriteImage(sitk.GetImageFromArray(images.cpu()), img_path)
    
    with torch.no_grad():
        mean_acc = torch.mean(torch.FloatTensor(acc_vals_eval))
        mean_loss = torch.mean(torch.FloatTensor(loss_vals_eval))
        mean_hr = torch.mean(torch.FloatTensor((torch.cat(hr_vals_eval))))
        mean_ccl = torch.mean(torch.FloatTensor(torch.cat((ccl_vals_eval))))
        mean_dist = torch.mean(torch.FloatTensor(np.concatenate(dist_vals_eval)))
        mse_x_epoch = torch.mean(torch.FloatTensor(np.stack(mse_x)))
        mse_y_epoch = torch.mean(torch.FloatTensor(np.stack(mse_y)))
        mse_z_epoch = torch.mean(torch.FloatTensor(np.stack(mse_z)))

    return mean_loss, mean_acc, mean_hr, mean_ccl, mean_dist, mse_x_epoch, mse_y_epoch, mse_z_epoch


def train_pertimestep_subsample(model, agent, train_ds_dict, val_ds_dict, num_epochs = 10, use_cuda = False, save_folder = 'model_1', loss_fn_str = 'MSE', clip_actions = False, num_sub_epochs = 100):
    
    """
    A function that performs the training and validation loop 
    :params:
    :prostate_dataloader: torch dataloader that can load images and masks in
    :num_epochs: (int) Number of epochs to train for 
    """
    current_dir = os.getcwd()
    train_folder = os.path.join(current_dir, save_folder)

    os.makedirs(train_folder, exist_ok = True) 
    print(f'train folder path : {train_folder}')
    writer = SummaryWriter(os.path.join(train_folder, 'runs')) 

    if use_cuda:
        model.cuda()
    
    # Defining optimiser, loss fn 
    #loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.MSELoss()
    print('loss fn : MSE Loss')

    data_mode = ['c2l', 'n2n', 'l2l', 'e2e']

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-05)

    # Parameters 
    step = 0 
    freq_print = 4
    freq_eval = 4
    sub_epochs = num_sub_epochs # num of sub epochs set to 100 by default 

    # Saving files 
    all_loss_train = np.zeros((num_epochs,1))
    all_acc_train = np.zeros((num_epochs, 1))
    all_loss_val = [] 
    all_acc_val = []
    best_loss = np.inf 
    best_acc = 0 
    
    csv_train_path = os.path.join(train_folder, 'train_loss.csv')
    csv_val_path = os.path.join(train_folder, 'val_loss.csv')

    with open(csv_train_path, 'w') as fp:
        fp.write('''epoch_num, bce_loss''')
        fp.write('\n')

    with open(csv_val_path, 'w') as fp:
        fp.write('''epoch_num, bce_loss''')
        fp.write('\n')

    # Initialise train and val dataset  
    mode = random.choice(data_mode)
    train_ds = train_ds_dict[mode]
    val_ds = val_ds_dict[mode]

    print(f"Using {mode} data for training")
    train_dataloader = DataLoader(train_ds, batch_size = 32, shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 8, shuffle = False)
    
    for epoch_no in range(num_epochs):
        
        print(f"\n Epoch number : {epoch_no}")
        acc_vals = []
        loss_vals = [] 
        hr_vals = [] 
        ccl_vals = []
        dist_vals = [] 

        # change training data every sub_epochs 
        if epoch_no % sub_epochs: 
            mode = random.choice(data_mode)
            print(f"\n Changing to data : {mode} for epoch num {epoch_no}")
            train_ds = train_ds_dict[mode]
            val_ds = val_ds_dict[mode]
            train_dataloader = DataLoader(train_ds, batch_size = 32, shuffle = True)
            val_dataloader = DataLoader(val_ds, batch_size = 8, shuffle = False)
                
        model.train()
        # Initialise training loop
        for idx, (images, labels, lesion_mask, action_identifier, tumour_centroid, grid_pos, max_depth) in enumerate(train_dataloader):
            
        #print(f'\n Idx train : {idx}')
            lesion_projection = np.max(lesion_mask.detach().numpy(), axis = 3)

            # Move to GPU 
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()

            # Training steps 
            optimiser.zero_grad()
            pred_actions, _, _ = model(images)  # Obtain predicted segmentation masks 

            # Clamp actions between -1,1 as done with labelled examples 
            if clip_actions:
                pred_actions = torch.clamp(pred_actions, -1, 1)
            
            loss = loss_fn(pred_actions.float(), labels.float()) # Compute the masks 
            loss.backward() # Backward propagation of gradients with respect to loss 
            optimiser.step() 

            # Debugging print functions
            #print(f'Unique vals : {np.unique(images.detach().cpu())}')
            #print(f'Unique vals : {np.unique(pred_masks.detach().cpu())}')
            #print(f'Unique vals label: {np.unique(labels.detach().cpu())}')
            #print(f'Loss : {loss.item()}')

            # Compute metrics for each mini batch and append to statistics for each epoch
            with torch.no_grad():
                accuracy = compute_rmse(loss)
            #    accuracy, precision, recall  = compute_accuracy(pred_actions[:,:,0:13, 0:13, :], labels)
            #    print(f'RMSE : {accuracy.item()}')
                acc_vals.append(accuracy)

                hit_rate, ccl = compute_HR(images, pred_actions, grid_pos, max_depth) #, lesion_projection, lesion_mask)
                #dist, mean_dist  = compute_dist_to_lesion(images, pred_actions, action_identifier, tumour_centroid, lesion_mask)
                dist, mean_dist = compute_DIST(images, pred_actions, tumour_centroid)
                
                # Only add HR and CCL for fired positions; not for non-fired 
                if np.all(hit_rate != None):
                    hr_vals.append(hit_rate)
                if np.all(ccl != None):
                    ccl_vals.append(ccl)
                
                dist_vals.append(dist)

            loss_vals.append(loss.item())

            # Print loss every nth minibatch and dice score 
            #if idx % freq_print == 0: 
            #    print(f'Epoch {epoch_no} minibatch {idx} : loss : {loss.item():05f}, acc score : {acc.item():05f}')
            
        # Obtain mean dice loss and acc over this epoch, save to tensorboard
        with torch.no_grad():
            acc_epoch = torch.mean(torch.tensor((acc_vals)))
            loss_epoch = torch.mean(torch.tensor((loss_vals)))

            # metrics 
            hr_epoch = torch.mean(torch.tensor(np.concatenate(hr_vals)).float())
            ccl_epoch = torch.mean(torch.tensor(np.concatenate(ccl_vals)).float())
            dist_epoch = torch.mean(torch.tensor(np.concatenate(dist_vals)).float())

            # compute std 
            hr_std = torch.mean(torch.tensor(np.concatenate(hr_vals)).float())
            ccl_std = torch.mean(torch.tensor(np.concatenate(ccl_vals)).float())
            dist_std = torch.mean(torch.tensor(np.concatenate(dist_vals)).float())


        print(f'\n Epoch : {epoch_no} Average loss : {loss_epoch:5f} average RMSE {acc_epoch:5f}')
        print(f'Metrics : HR {hr_epoch} +- {hr_std}, CCL : {ccl_epoch} +- {ccl_std}, dist metric : {dist_epoch} +- {dist_std}')

        with open(csv_train_path, 'a') as fp: 
            loss_points = np.stack([epoch_no, loss_epoch]).reshape(1,-1)
            np.savetxt(fp, loss_points, '%s', delimiter =",")

        # Save for all_loss_train
        all_loss_train[epoch_no] = loss_epoch
        all_acc_train[epoch_no] = acc_epoch 
        
        #Tensorboard saving 
        writer.add_scalar('Loss/train', loss_epoch, epoch_no)
        writer.add_scalar('RMSE/train', acc_epoch, epoch_no)
        writer.add_scalar('HR/train', hr_epoch, epoch_no)
        writer.add_scalar('CCL/train', ccl_epoch, epoch_no)
        writer.add_scalar('dist/train', dist_epoch, epoch_no)
    
        # Save newest model 
        train_model_path = os.path.join(train_folder, 'train_model.pth')
        torch.save(model.state_dict(), train_model_path)

        # Validate every nth epoch and save every nth mini batch 
        
        if epoch_no % freq_eval == 0: 
            print("VALIDATION : \n ")
            model.eval()        
            mean_loss, mean_acc, mean_hr, mean_ccl, mean_dist = validate_pertimestep_new(val_dataloader, model, use_cuda = use_cuda, save_path = train_folder, save_images = False, clip_actions = clip_actions)
            print(f'Validation loss for epoch {epoch_no} Average loss : {mean_loss:5f} average acc {mean_acc:5f}')
            all_loss_val.append(mean_loss)
            all_acc_val.append(mean_acc)
            
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, mean_loss, mean_acc]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

            #Tensorboard saving
            writer.add_scalar('Loss/val', mean_loss, epoch_no)
            writer.add_scalar('RMSE/val', mean_acc, epoch_no)
            writer.add_scalar('mean_HR/val', mean_hr, epoch_no)
            writer.add_scalar('mean_ccl/val', mean_ccl, epoch_no)
            writer.add_scalar('mean_dist/val', mean_dist, epoch_no)

            if mean_loss < best_loss: 
                
                # Save best model as best validation model 
                val_model_path = os.path.join(train_folder, 'best_val_model.pth')
                torch.save(model.state_dict(), val_model_path)
        
                # Use as new best loss
                best_loss = mean_loss 

                # Save agent
                agent.policy = model 
                agent.save(os.path.join(save_folder, "agent_model"))
        
        elif epoch_no == 0: 
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, 1]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")
        else:
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, all_loss_val[-1]]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

        #print('Chicken') 
    return all_loss_train, all_loss_val, all_acc_train, all_acc_val 

def train_pertimestep_sequential(model, agent, train_ds_dict, val_ds_dict, num_epochs = 2000, use_cuda = False, save_folder = 'model_1', loss_fn_str = 'MSE', clip_actions = False, num_sub_epochs = 100):
    
    """
    A function that performs the training and validation loop 
    :params:
    :prostate_dataloader: torch dataloader that can load images and masks in
    :num_epochs: (int) Number of epochs to train for 
    """
    current_dir = os.getcwd()
    train_folder = os.path.join(current_dir, save_folder)

    os.makedirs(train_folder, exist_ok = True) 
    print(f'train folder path : {train_folder}')
    writer = SummaryWriter(os.path.join(train_folder, 'runs')) 

    if use_cuda:
        model.cuda()
    
    # Defining optimiser, loss fn 
    #loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.MSELoss()
    print('loss fn : MSE Loss')

    data_mode = ['c2l', 'n2n', 'l2l', 'e2e']

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-05)

    # Parameters 
    step = 0 
    freq_print = 4
    freq_eval = 4
    sub_epochs = num_sub_epochs # num of sub epochs set to 100 by default 

    # Saving files 
    all_loss_train = np.zeros((num_epochs,1))
    all_acc_train = np.zeros((num_epochs, 1))
    all_loss_val = [] 
    all_acc_val = []
    best_loss = np.inf 
    best_acc = 0 
    
    csv_train_path = os.path.join(train_folder, 'train_loss.csv')
    csv_val_path = os.path.join(train_folder, 'val_loss.csv')

    with open(csv_train_path, 'w') as fp:
        fp.write('''epoch_num, bce_loss''')
        fp.write('\n')

    with open(csv_val_path, 'w') as fp:
        fp.write('''epoch_num, bce_loss''')
        fp.write('\n')

    # Initialise train and val dataset  at c2l for first 500 epochs 
    mode = 'c2l'
    train_ds = train_ds_dict[mode]
    val_ds = val_ds_dict[mode]

    print(f"Using {mode} data for training")
    train_dataloader = DataLoader(train_ds, batch_size = 32, shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 8, shuffle = False)
    
    # Train for 1500 epochs only 
    c2l_epochs = 500
    n2n_epochs = 1000 # 500 < epoch_no <= 1000
    l2l_epochs = 1500 # 1000 < epoch_no <= 1500 
    e2e_epochs = 2000 # 1500 < epoch_no < 2000 

    for epoch_no in range(num_epochs):
        
        print(f"\n Epoch number : {epoch_no}")
        acc_vals = []
        loss_vals = [] 
        hr_vals = [] 
        ccl_vals = []
        dist_vals = [] 

        # Change ds at epoch_no == 500 to n2n 
        if epoch_no == 501:
            mode = 'n2n' 
            print(f"Epoch num : {epoch_no} changing dataset to {mode}")
            train_ds = train_ds_dict[mode]
            val_ds = val_ds_dict[mode]
            train_dataloader = DataLoader(train_ds, batch_size = 32, shuffle = True)
            val_dataloader = DataLoader(val_ds, batch_size = 8, shuffle = False)
        
        # change ds at epoch_no == 10001 to l2l 
        elif epoch_no == 1001: 
            mode = 'l2l'
            print(f"Epoch num : {epoch_no} changing dataset to {mode}")
            train_ds = train_ds_dict[mode]
            val_ds = val_ds_dict[mode]
            train_dataloader = DataLoader(train_ds, batch_size = 32, shuffle = True)
            val_dataloader = DataLoader(val_ds, batch_size = 8, shuffle = False)

        # change ds at epoch_no == 1501 to e2e
        elif epoch_no == 1501: 
            mode = 'e2e'
            print(f"Epoch num : {epoch_no} changing dataset to {mode}")
            train_ds = train_ds_dict[mode]
            val_ds = val_ds_dict[mode]
            train_dataloader = DataLoader(train_ds, batch_size = 32, shuffle = True)
            val_dataloader = DataLoader(val_ds, batch_size = 8, shuffle = False)

        
        # # change training data every sub_epochs 
        # if epoch_no % sub_epochs: 
        #     mode = random.choice(data_mode)
        #     print(f"\n Changing to data : {mode} for epoch num {epoch_no}")
        #     train_ds = train_ds_dict[mode]
        #     val_ds = val_ds_dict[mode]
        #     train_dataloader = DataLoader(train_ds, batch_size = 32, shuffle = True)
        #     val_dataloader = DataLoader(val_ds, batch_size = 8, shuffle = False)
                
        model.train()
        # Initialise training loop
        for idx, (images, labels, lesion_mask, action_identifier, tumour_centroid, grid_pos, max_depth) in enumerate(train_dataloader):
            
        #print(f'\n Idx train : {idx}')
            lesion_projection = np.max(lesion_mask.detach().numpy(), axis = 3)

            # Move to GPU 
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()

            # Training steps 
            optimiser.zero_grad()
            pred_actions, _, _ = model(images)  # Obtain predicted segmentation masks 

            # Clamp actions between -1,1 as done with labelled examples 
            if clip_actions:
                pred_actions = torch.clamp(pred_actions, -1, 1)
            
            loss = loss_fn(pred_actions.float(), labels.float()) # Compute the masks 
            loss.backward() # Backward propagation of gradients with respect to loss 
            optimiser.step() 

            # Debugging print functions
            #print(f'Unique vals : {np.unique(images.detach().cpu())}')
            #print(f'Unique vals : {np.unique(pred_masks.detach().cpu())}')
            #print(f'Unique vals label: {np.unique(labels.detach().cpu())}')
            #print(f'Loss : {loss.item()}')

            # Compute metrics for each mini batch and append to statistics for each epoch
            with torch.no_grad():
                accuracy = compute_rmse(loss)
            #    accuracy, precision, recall  = compute_accuracy(pred_actions[:,:,0:13, 0:13, :], labels)
            #    print(f'RMSE : {accuracy.item()}')
                acc_vals.append(accuracy)

                hit_rate, ccl = compute_HR(images, pred_actions, grid_pos, max_depth) #, lesion_projection, lesion_mask)
                #dist, mean_dist  = compute_dist_to_lesion(images, pred_actions, action_identifier, tumour_centroid, lesion_mask)
                dist, mean_dist = compute_DIST(images, pred_actions, tumour_centroid)
                
                # Only add HR and CCL for fired positions; not for non-fired 
                if np.all(hit_rate != None):
                    hr_vals.append(hit_rate)
                if np.all(ccl != None):
                    ccl_vals.append(ccl)
                
                dist_vals.append(dist)

            loss_vals.append(loss.item())

            # Print loss every nth minibatch and dice score 
            #if idx % freq_print == 0: 
            #    print(f'Epoch {epoch_no} minibatch {idx} : loss : {loss.item():05f}, acc score : {acc.item():05f}')
            
        # Obtain mean dice loss and acc over this epoch, save to tensorboard
        with torch.no_grad():
            acc_epoch = torch.mean(torch.tensor((acc_vals)))
            loss_epoch = torch.mean(torch.tensor((loss_vals)))

            # metrics 
            hr_epoch = torch.mean(torch.tensor(np.concatenate(hr_vals)).float())
            ccl_epoch = torch.mean(torch.tensor(np.concatenate(ccl_vals)).float())
            dist_epoch = torch.mean(torch.tensor(np.concatenate(dist_vals)).float())

            # compute std 
            hr_std = torch.mean(torch.tensor(np.concatenate(hr_vals)).float())
            ccl_std = torch.mean(torch.tensor(np.concatenate(ccl_vals)).float())
            dist_std = torch.mean(torch.tensor(np.concatenate(dist_vals)).float())


        print(f'\n Epoch : {epoch_no} Average loss : {loss_epoch:5f} average RMSE {acc_epoch:5f}')
        print(f'Metrics : HR {hr_epoch} +- {hr_std}, CCL : {ccl_epoch} +- {ccl_std}, dist metric : {dist_epoch} +- {dist_std}')

        with open(csv_train_path, 'a') as fp: 
            loss_points = np.stack([epoch_no, loss_epoch]).reshape(1,-1)
            np.savetxt(fp, loss_points, '%s', delimiter =",")

        # Save for all_loss_train
        all_loss_train[epoch_no] = loss_epoch
        all_acc_train[epoch_no] = acc_epoch 
        
        #Tensorboard saving 
        writer.add_scalar('Loss/train', loss_epoch, epoch_no)
        writer.add_scalar('RMSE/train', acc_epoch, epoch_no)
        writer.add_scalar('HR/train', hr_epoch, epoch_no)
        writer.add_scalar('CCL/train', ccl_epoch, epoch_no)
        writer.add_scalar('dist/train', dist_epoch, epoch_no)
    
        # Save newest model 
        train_model_path = os.path.join(train_folder, 'train_model.pth')
        torch.save(model.state_dict(), train_model_path)

        # Validate every nth epoch and save every nth mini batch 
        
        if epoch_no % freq_eval == 0: 
            print("VALIDATION : \n ")
            model.eval()        
            mean_loss, mean_acc, mean_hr, mean_ccl, mean_dist, mse_x_epoch, mse_y_epoch, mse_z_epoch = validate_pertimestep_new(val_dataloader, model, use_cuda = use_cuda, save_path = train_folder, save_images = False, clip_actions = clip_actions)
            print(f'Validation loss for epoch {epoch_no} Average loss : {mean_loss:5f} average acc {mean_acc:5f}')
            all_loss_val.append(mean_loss)
            all_acc_val.append(mean_acc)
            
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, mean_loss, mean_acc]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

            #Tensorboard saving
            writer.add_scalar('Loss/val', mean_loss, epoch_no)
            writer.add_scalar('RMSE/val', mean_acc, epoch_no)
            writer.add_scalar('mean_HR/val', mean_hr, epoch_no)
            writer.add_scalar('mean_ccl/val', mean_ccl, epoch_no)
            writer.add_scalar('mean_dist/val', mean_dist, epoch_no)
            writer.add_scalar('mse_x/train', mse_x_epoch, epoch_no)
            writer.add_scalar('mse_y/train', mse_y_epoch, epoch_no)
            writer.add_scalar('mse_z/train', mse_z_epoch, epoch_no)

            if mean_loss < best_loss: 
                
                # Save best model as best validation model 
                val_model_path = os.path.join(train_folder, 'best_val_model.pth')
                torch.save(model.state_dict(), val_model_path)
        
                # Use as new best loss
                best_loss = mean_loss 

                # Save agent
                agent.policy = model 
                agent.save(os.path.join(save_folder, "agent_model"))
        
        elif epoch_no == 0: 
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, 1]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")
        else:
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, all_loss_val[-1]]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

        #print('Chicken') 
    return all_loss_train, all_loss_val, all_acc_train, all_acc_val 

def compute_HR_CCL(grid_obs, actions, lesion_projection, lesion_mask):
    """
    Computes hit rate, given observations and actions 

    Parameters:
    ------------
    :grid_obs: batch_size x [100 x 100 x 75]
    :actions: batch_size x [1 x 3]
    :lesion_projection: batch_size x [100 x 100]


    Returns:
    -----------
    :hit_rate: float computed by number_hit / number_fired 
    :ccl_coeff: coefficient between ccl and lesion size 
    """
    # swap actions such that x is second, y is first dim 
    swapped_actions = copy.deepcopy(actions[:, [1,0,2]]).cpu()

    # if action[-1] == 1 (ie needle fired): 
    # array of hits (ie needle fired)

    bool_idx = swapped_actions[:, -1] > 0 # if the final action value is >0 then fired, otherwise non-fired
    fired_actions = swapped_actions[bool_idx]
    NUM_ACTIONS = len(fired_actions)

    # compute new observation, as a result of action 
    fired_obs = torch.squeeze(grid_obs[bool_idx, :, :, :, 50]).cpu()

    # where is current grid_pos? 
    if NUM_ACTIONS == 0: # ie no fired grid, no HR to compute or CCL
        hit_rate = None 
        ccl_array = None 

        print(f'No needles fired, HR and CCL = None')
        return hit_rate, ccl_array

    elif NUM_ACTIONS == 1: 
        current_grid_pos = torch.tensor(np.mean(np.where(fired_obs == 0.25), axis = 1))
    else:
        current_grid_pos = torch.tensor([np.mean(np.where(fired_obs[idx] == 0.25), axis = 1) for idx in range(NUM_ACTIONS)])


    # find new grid pos, as a result of the action 
    scaled_actions = (fired_actions[:, :-1] * 5 * 2) # only using x, y; multiply by 5 to get 5mm intervals; multiply by 2 because previously normalised by dividing by 2 so unormalising now 
    new_grid_pos = (scaled_actions + current_grid_pos).int() 

    # check whether grid position intersects with lesion mask 
    hit_array = np.zeros(NUM_ACTIONS)
    ccl_array = np.zeros(NUM_ACTIONS)

    for idx, coords in enumerate(new_grid_pos): 

        #print(coords)

        # Compute hit rate by checking if needle coord intersects with lesion 
        intensities = lesion_projection[idx, coords[0]-1:coords[0]+2, coords[1]-1:coords[1]+2] #checks neighbouring values (assumes 3mm diameter of needle)
        intersects = np.any(intensities == 1)
        #print(intersects)
        hit_array[idx] = intersects

        # compute CCL: ie maximum length (1 * 4mm) only for fired coords!!
        needle_traj = lesion_mask[idx, coords[0]-1:coords[0]+2, coords[1]-1:coords[1]+2, :]
        z_vals = np.where(needle_traj)[-1] # check where intensities are 1 ie hit lesion
        
        if len(z_vals) == 0: 
            ccl = 0 # no intersection
        else: 
            # *4 to upsample volume in mm 
            ccl = 4*(np.max(z_vals) - np.min(z_vals))# crude estimate of ccl : max_z - min_z (ie length of z covered) * 4 because we previous downsampled

        #print(f'CCL : {ccl}')
        ccl_array[idx] = ccl 

    # compute hit rate: 
        # HR = num_intersections / num_fired_needles 
    
    # separately, compute CCL coeff -> next time!!! 

    # compute hit rate 
    hit_rate = np.mean(hit_array) * 100
    std_hr = np.std(hit_array)
    mean_ccl = np.mean(ccl_array)
    std_ccl = np.std(ccl_array)
    
    # print statistics:
    print(f'Hit rate : {hit_rate} +- {std_hr} for N = {NUM_ACTIONS} \n')
    print(f'CCL : {mean_ccl} +- {std_ccl} for N = {NUM_ACTIONS}')

    return hit_array, ccl_array 

def compute_dist_to_lesion(grid_obs, actions, action_identifier, tumour_centroid, lesion_projection):
    """
    Computes distance to closest lesion and whether agents' step is closer or further away than current position 
    TODO: add index of lesion targeted by action to action labels!!! to use for compute_dist_to_lesion metric 
    
    """

    # Check which lesion the current action is meant to target 
    obs = copy.deepcopy(torch.squeeze(grid_obs[:, :, :, :, 50])).cpu()
    NUM_OBS = np.shape(obs)[0]

    if NUM_OBS == 1: 
        current_grid_pos = torch.tensor(np.mean(np.where(obs == 0.25), axis = 1))
    else:
        current_grid_pos = torch.tensor([np.mean(np.where(obs[idx] == 0.25), axis = 1) for idx in range(NUM_OBS)])

    # Check distance between current position and target lesion 
    # find new grid pos, as a result of the action 
    swapped_actions = copy.deepcopy(actions[:, [1,0,2]]).cpu()
    scaled_actions = (swapped_actions[:, :-1] * 5 * 2) # only using x, y; multiply by 5 to get 5mm intervals; multiply by 2 because previously normalised by dividing by 2 so unormalising now 
    new_grid_pos = (scaled_actions + current_grid_pos).int()  # y x z 

    # Compute distance between new position (current + action) and target lesion 
    # change tumour_centroid from
    swapped_centres = tumour_centroid[:, [1,0,2]]

    # compute distance for each row 
    dist_t1 = np.linalg.norm(swapped_centres[:,0:2] - new_grid_pos, axis = 1) 
    dist_t0 = np.linalg.norm(swapped_centres[:,0:2] - current_grid_pos, axis = 1) 

    # sign : (dist_t+1 - dist_t) if +ve -> moving closer. If not, moving further away! --> gives indication of gradient. 
    sign_dist = np.sign(dist_t0 - dist_t1) # if t0 > t1 -> getting closer to centroid 
    #print(f'Sign(dist_t - dist_t+1 : {sign_dist} average : {np.mean(sign_dist)} +- {np.std(sign_dist)}')

    return sign_dist, np.mean(sign_dist)

class NewFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        Correspods to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, multiple_frames = False, num_channels = 5):
        
        super(NewFeatureExtractor, self).__init__(observation_space, features_dim)
        # Assumes CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        #num_input_channels = observation_space.shape[-1] #rows x cols x channels 
        #num_multiple_frames = 3
        #num_multiple_frames = observation_space.shape[-1]
        #self.num_multiple_frames = num_multiple_frames

        num_channels = num_channels
        self.cnn_layers = nn.Sequential(

            # First layer like resnet, stride = 2
            nn.Conv3d(num_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            # Apply pooling layer in between 
            nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            # Apply pooling layer in between 
            nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1),

            nn.Conv3d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        #Flatten layers 
        self.flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            all_layers = nn.Sequential(self.cnn_layers, self.flatten)
            
            #observation_space_shuffled = np.transpose(observation_space.sample(), [2, 1, 0])
            #n_flatten = all_layers(torch.as_tensor(observation_space_shuffled[None]).float()).shape[1]
            #processed_obs_space = self._pre_process_image(torch.zeros))).float()
            processed_obs_space = torch.zeros([1, num_channels, 100, 100, 24])
            n_flatten = all_layers(processed_obs_space).shape[1]  

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        #observations = self._pre_process_image(observations)
        observations = observations.float() 
        output = self.cnn_layers(observations)
        output = self.flatten(output)
        
        return self.linear(output)

    def _pre_process_image(self, images):
        """ 
        A function that switches the dimension of image from row x col x channel -> channel x row x colmn 
        and addeds a dimension along 0th axis to fit network 
        """ 
        #print(f'Image size {images.size()}')
        image = images.clone().detach().to(torch.uint8)#.squeeze()
        if len(np.shape(images)) == 5:
            image = image.squeeze()
        split_channel_image = torch.cat([torch.cat([image[j,:,:,i*25:(i*25)+25].unsqueeze(0) for i in range(3)]).unsqueeze(0) for j in range(image.size()[0])])#.clone().detach().to(torch.uint8)
        #split_channel_image = torch.cat([torch.cat(torch.tensor_split(image[i,:,:,:].unsqueeze(0), self.num_multiple_frames, dim=3)).unsqueeze(0) for i in range(image.size()[0])])
        #processed_image = image.permute(0, 3,2,1)
        #processed_image = torch.unsqueeze(processed_image, dim= 0)
        
        # Turn image from channel x row x column -> channel x row x column x depth for pre-processing with 3D layers 

        return split_channel_image
if __name__ == '__main__':

    # Test functions used for training 
    
    pass 