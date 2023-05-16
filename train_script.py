import numpy as np 
from matplotlib import pyplot as plt 
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader, RandomSampler 
import pandas as pd 
import os 
import h5py
import argparse 
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.ppo.policies import CnnPolicy
from supersuit import frame_stack_v1
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gym
#Processes for multi-env processing 
from rl_utils import *
from utils import *
from stable_baselines3.common.policies import ActorCriticPolicy

#Importing module functions 
from Prostate_dataloader import *
from Biopsy_env import TemplateGuidedBiopsy_penalty

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

parser.add_argument('--clip_actions',
                    '--clip',
                    metavar='clip_actions',
                    type=bool,
                    action='store',
                    default=False,
                    help='Whether or not to clip actions during training /testing')

parser.add_argument('--use_custom_policy',
                    '--custom',
                    metavar='use_custom_policy',
                    type=bool,
                    action='store',
                    default=False,
                    help='Whether or not to use custom policy for networks')

parser.add_argument('--debugging',
                    '--debug',
                    metavar='debugging',
                    type=bool,
                    action='store',
                    default=False,
                    help='Whether or not to debug using same actions only')

class TimeStep_data(Dataset):

    def __init__(self, folder_name, csv_path, labels_path = 'action_labels.h5', mode = 'train', finetune = False):

        self.folder_name = folder_name
        self.mode = mode
        self.labels_path = labels_path
        self.finetune = finetune

        # Obtain list of patient names with multiple lesions -> change to path name
        #csv_path ='/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv'
        df_dataset = pd.read_csv(csv_path)
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
        """ normalise actions in the range (-1,1)"""

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

class TimeStep_data_debug(Dataset):
    """
    Debugging dataloader : always returns the same action [1 0 0] ie always going in a horizontal striaghtg line -> 
    """

    def __init__(self, folder_name, csv_path, labels_path = 'action_labels.h5', mode = 'train', finetune = False):

        self.folder_name = folder_name
        self.mode = mode
        self.labels_path = labels_path
        self.finetune = finetune

        # Obtain list of patient names with multiple lesions -> change to path name
        df_dataset = pd.read_csv(csv_path)
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

def compute_rmse(mse):
    """
    Computes rmse as a metric to be reported, but not to be used for comptuation of loss fns 
    """

    with torch.no_grad():
        mse_copy = torch.clone(mse.detach())
        rmse = torch.sqrt(mse_copy)

    return rmse 

def train_pertimestep_old(model, agent, train_dataloader, val_dataloader, num_epochs = 10, use_cuda = False, save_folder = 'model_1', loss_fn_str = 'MSE', clip_actions = False):
    
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

        model.train()
        # Initialise training loop
        for idx, (images, labels) in enumerate(train_dataloader):
            
        #print(f'\n Idx train : {idx}')

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
            
            loss_vals.append(loss.item())

            # Print loss every nth minibatch and dice score 
            #if idx % freq_print == 0: 
            #    print(f'Epoch {epoch_no} minibatch {idx} : loss : {loss.item():05f}, acc score : {acc.item():05f}')
            
        # Obtain mean dice loss and acc over this epoch, save to tensorboard
        with torch.no_grad():
            acc_epoch = torch.mean(torch.tensor(acc_vals))
            loss_epoch = torch.mean(torch.tensor(loss_vals))

        print(f'\n Epoch : {epoch_no} Average loss : {loss_epoch:5f} average RMSE {acc_epoch:5f}')

        with open(csv_train_path, 'a') as fp: 
            loss_points = np.stack([epoch_no, loss_epoch]).reshape(1,-1)
            np.savetxt(fp, loss_points, '%s', delimiter =",")

        # Save for all_loss_train
        all_loss_train[epoch_no] = loss_epoch
        all_acc_train[epoch_no] = acc_epoch 
        
        #Tensorboard saving 
        writer.add_scalar('Loss/train', loss_epoch, epoch_no)
        writer.add_scalar('RMSE/train', acc_epoch, epoch_no)
    
        # Save newest model 
        train_model_path = os.path.join(train_folder, 'train_model.pth')
        torch.save(model.state_dict(), train_model_path)

        # Validate every nth epoch and save every nth mini batch 
        
        if epoch_no % freq_eval == 0: 
            print("VALIDATION : \n ")
            model.eval()        
            mean_loss, mean_acc = validate_pertimestep(val_dataloader, model, use_cuda = use_cuda, save_path = train_folder, save_images = False, clip_actions = clip_actions)
            print(f'Validation loss for epoch {epoch_no} Average loss : {mean_loss:5f} average acc {mean_acc:5f}')
            all_loss_val.append(mean_loss)
            all_acc_val.append(mean_acc)
            
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, mean_loss, mean_acc]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

            #Tensorboard saving
            writer.add_scalar('Loss/val', mean_loss, epoch_no)
            writer.add_scalar('RMSE/val', mean_acc, epoch_no)

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

def validate_pertimestep_old(val_dataloader, model, use_cuda = True, save_path = 'model_1', save_images = False, metric = 'rmse', clip_actions = False):

    # Set to evaluation mode 
    model.eval()
    acc_vals_eval = [] 
    loss_vals_eval = [] 

    loss_fn = torch.nn.BCEWithLogitsLoss()
    for idx, (images, labels) in enumerate(val_dataloader):
        
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

        #if save_images:
        #    # Save image, labels and outputs into h5py files
        #    img_name = patient_name[0].split(".")[0] + '_rectum_PRED.nrrd'
        #    img_path = os.path.join(save_path, img_name)
        #    sitk.WriteImage(sitk.GetImageFromArray(images.cpu()), img_path)
    
    with torch.no_grad():
        mean_acc = torch.mean(torch.FloatTensor(acc_vals_eval))
        mean_loss = torch.mean(torch.FloatTensor(loss_vals_eval))

    return mean_loss, mean_acc

class TimeStep_data_moreinfo(Dataset):

    def __init__(self, folder_name, labels_path = 'action_labels.h5', mode = 'train', finetune = False, single_patient = False):

        self.folder_name = folder_name
        self.mode = mode
        self.labels_path = labels_path
        self.finetune = finetune
        self.single_patient = single_patient
        
        # Obtain list of patient names with multiple lesions -> change to path name
        df_dataset = pd.read_csv('/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv')
        #df_dataset = pd.read_csv('/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv')
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
            needle_stack = torch.tensor([np.zeros([100,100,24]), np.zeros([100,100,24]), needle_vol[random_idx]])
        elif random_idx == 1:
            #sampled_grid = np.array([np.zeros((100,100)), np.array(all_grids[random_idx-1]), np.array(all_grids[random_idx])])
            needle_stack = torch.tensor([np.zeros([100,100,24]), needle_vol[random_idx-1], needle_vol[random_idx]])
        else:
            #sampled_grid = all_grids[random_idx - 2 :random_idx+1 , :]
            needle_stack = torch.tensor(needle_vol[random_idx-2 : random_idx+1])
            
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
        
        return obs_stack, final_action, downsampled_lesion_mask, action_identifier, tumour_centroid
    
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

### NEW TRAINING FUNCTIONS
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

                hit_rate, ccl = compute_HR(images, pred_actions) #, lesion_projection, lesion_mask)
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
            hr_epoch = torch.mean(torch.tensor(np.concatenate(hr_vals)))
            ccl_epoch = torch.mean(torch.tensor(np.concatenate(ccl_vals)).float())
            dist_epoch = torch.mean(torch.tensor(np.concatenate(dist_vals)))

            # compute std 
            hr_std = torch.mean(torch.tensor(np.concatenate(hr_vals)))
            ccl_std = torch.mean(torch.tensor(np.concatenate(ccl_vals)).float())
            dist_std = torch.mean(torch.tensor(np.concatenate(dist_vals)))


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
            mean_loss, mean_acc, mean_hr, mean_ccl, mean_dist = validate_pertimestep(val_dataloader, model, use_cuda = use_cuda, save_path = train_folder, save_images = False, clip_actions = clip_actions)
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


            hit_rate, ccl = compute_HR(images, labels) #, lesion_projection, lesion_mask)
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

    return mean_loss, mean_acc, mean_hr, mean_ccl, mean_dist 

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
    print(f'Sign(dist_t - dist_t+1 : {sign_dist} average : {np.mean(sign_dist)} +- {np.std(sign_dist)}')

    return sign_dist, np.mean(sign_dist)
     
def compute_DIST(obs, actions, tumour_centroid):
    """
    Computes distance to closest lesion and whether agents' step is closer or further away than current position 
    TODO: add index of lesion targeted by action to action labels!!! to use for compute_dist_to_lesion metric 
    
    """
    raw_actions = torch.clamp(actions, -1,1)
    raw_actions = actions.detach().numpy()

    needle_mask = obs[:,-1,:,:,0]
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
    print(f'Sign(dist_t - dist_t+1 : {sign_dist} average : {np.mean(sign_dist)} +- {np.std(sign_dist)}')

    return sign_dist, np.mean(sign_dist)
     
def compute_HR(obs, actions):
    """
    Compute both HR and CCL using observations 

    obs : (batch_size x 5 x 100 x 100 x 24)

    """

    # clip actions between (-1,1)
    pred_actions = torch.clamp(actions, -1, 1)

    lesion = obs[:,0,:,:,:] 
    needle = obs[:,4,:,:,:]

    # find which idx are fired
    fired_idx = np.where(pred_actions[:,-1] >= -0.33)[0]
    intersect = lesion[fired_idx, :, :, :] * needle[fired_idx,  :, :, :]
    NUM_FIRED = len(fired_idx)

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

# Networks to use for training 
class FeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        Correspods to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, multiple_frames = False, num_multiple_frames = 3):
        
        super(FeatureExtractor, self).__init__(observation_space, features_dim)
        # Assumes CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        #num_input_channels = observation_space.shape[-1] #rows x cols x channels 
        #num_multiple_frames = 3
        #num_multiple_frames = observation_space.shape[-1]
        #self.num_multiple_frames = num_multiple_frames

        num_channels = 5 
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
            processed_obs_space = torch.zeros([1, 5, 100, 100, 24])
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

class CustomNetwork(nn.Module):

    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.Tanh()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.Tanh()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

if __name__ =='__main__':
    
    ### Path files for PT
    args = parser.parse_args()

    # Change to paths on your device 
    PS_PATH = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality'
    LABELS_PATH = '/Users/ianijirahmae/ImitationLearning/ACTION_OBS_LABELS.h5'
    CSV_PATH = '/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv'

    # Training process parameters    
    TRAIN_MODE = 'train'
    LOG_DIR = args.log_dir  # folder path to save trained models into 
    use_cuda = torch.cuda.is_available()
    device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device_cuda}')

    # Training process parameters : all set to false by default. 
    use_custom_policy = args.use_custom_policy # custom policy uses tanh instead of Relu for final layer of network 
    clip_actions = args.clip_actions # clip actions between (-1,1) during inference 
    DEBUGGING = args.debugging # whether or not to use debugging datalodaer 

    # for info printing 
    print(f'Clipped actions : {clip_actions} ; using custom policy with tanh : {use_custom_policy}')
    print(f"Using debugging dataloader: {DEBUGGING}")

    # Setting up RL networks to extract custom policy networks used for RL training 
    PS_dataset = Image_dataloader(PS_PATH, CSV_PATH, use_all = True, mode  = TRAIN_MODE)
    Data_sampler= DataSampler(PS_dataset)
    Biopsy_env_init = TemplateGuidedBiopsy_penalty(Data_sampler, train_mode = TRAIN_MODE)
    Biopsy_env = frame_stack_v1(Biopsy_env_init, 3)
    Biopsy_env.reset()

    # Initialising agent to be trained 
    if use_custom_policy: 
        # uses custom policy for final layers of actor-critic network (ie tanh instead of relu)
        policy_kwargs = dict(features_extractor_class = FeatureExtractor, \
        features_extractor_kwargs=dict(multiple_frames = True, num_multiple_frames = 3), activation_fn = torch.nn.Tanh)
        agent = PPO(CustomActorCriticPolicy, Biopsy_env, policy_kwargs = policy_kwargs, tensorboard_log = LOG_DIR)
    else:  
        policy_kwargs = dict(features_extractor_class = FeatureExtractor, features_extractor_kwargs=dict(multiple_frames = True, num_multiple_frames = 3))  # defines feature extractor to be used for network 
        agent = PPO(CnnPolicy, Biopsy_env, policy_kwargs = policy_kwargs, device = device_cuda, tensorboard_log = LOG_DIR)

    # Use same policy network structure used by agents in RL 
    IL_MODEL = agent.policy.to(device_cuda)

    # Alternatively, initialise imitation learning network defined in networks.py 
    # IL_MODEL = ImitationNetwork()

    # Load paired observation-action time step data 
    if DEBUGGING: 
        # Uses debugging dataloader instead of normal data loader 
        # prints out patient idx and action idx to check for gradient updates
        train_ds = TimeStep_data_debug(PS_PATH, LABELS_PATH, mode = 'train')
        val_ds = TimeStep_data_debug(PS_PATH, LABELS_PATH, mode = 'val')
    else:
        train_ds = TimeStep_data_new(PS_PATH, CSV_PATH, LABELS_PATH, mode = 'train')
        val_ds = TimeStep_data_new(PS_PATH, CSV_PATH, LABELS_PATH, mode = 'val')

    train_dl = DataLoader(train_ds, batch_size = 32, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size = 8, shuffle = False)


    # Perform training loop : uses train_pertimestep function found in utils.py 
    all_loss_train, all_loss_val, all_acc_train, all_acc_val  = train_pertimestep(IL_MODEL, agent, train_dl, val_dl, \
        num_epochs = 10000, use_cuda = use_cuda, save_folder = LOG_DIR, clip_actions = clip_actions) 

    # Save agent 

    print('Chicken')