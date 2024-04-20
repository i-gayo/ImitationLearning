import numpy as np 
from matplotlib import pyplot as plt 
import SimpleITK as sitk
import os 
import pandas as pd 
import gym 
import torch
import torch.nn as nn 
import torchvision
from torchvision.models import resnet18 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.data import Dataset, DataLoader, RandomSampler 
import ast

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.logger import Figure
from utils.Prostate_dataloader import * 

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
                print(f"\n EVALUATING AVERAGE REWARD:  \
                      Best mean reward: {self.best_mean_reward:.2f} +/- {self.best_mean_reward_std:.2f} \
                - Last mean reward per episode: {mean_reward:.2f} +/- {std_reward:.2f}")

              # Save the model if the mean reward is better than the previously saved mean reward 
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  self.best_mean_reward_std = std_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True


class ImageReader:

    def __call__(self, file_path, require_sitk_img = False):
        image_vol = sitk.ReadImage(file_path)
        image_vol = sitk.GetArrayFromImage(image_vol)
        image_size = np.shape(image_vol)

        if require_sitk_img: 
            sitk_image_vol = sitk.ReadImage(file_path, sitk.sitkUInt8)
            return image_vol, sitk_image_vol

        else:        
            return image_vol

class RectumLabeller:

    def __init__(self, folder_path, csv_file = 'rectum_pos.csv'):
        self.folder_path = folder_path 
        self.all_file_names = [f for f in os.listdir(folder_path) if not f.startswith('.')]
        self.all_file_paths = [os.path.join(folder_path, file_name) for file_name in self.all_file_names]
        self.csv_file = csv_file 

        #Ensures a new line is started for CSV file 
        with open(self.csv_file, 'a') as f:
            f.write('\n')

    def save_to_csv():
        """ A function that saves all file names to csv, along with corresponding rectum positions """
        pass

    def display_figure():
        """
        A method that displays figure with first non-zero value 
        """
        pass

    def take_user_input(self):
        """
        A method that takes user input for x,y files after being shown MRI figure 
        """
        
        input_x = input("Enter rectum coords x : ")
        input_y = input("Enter rectum coords y : ")
        coords = [int(input_x), int(input_y)]
        print(f'Coords : {coords}')

        return coords

    def __call__(self, begin_idx =0 , end_idx = 3):
        """
        Labels rectums from idx = begin_idx up till end_idx at any given time 
        """

        #idx_vals = np.arange(begin_idx, end_idx)

        for i in range(begin_idx, end_idx): #idx_vals:

            #Load image 
            reader = ImageReader()
            mri_vol, mri_volsize = reader(self.all_file_paths[i])

            # Find index where first non-zero image is present
            unique_intensities = [len(np.unique(mri_vol[i,:,:])) for i in range(mri_volsize[0])]
            z = np.argwhere(np.array(unique_intensities) != 1)[0][0]

            # Display figure
            plt.figure()
            plt.imshow(mri_vol[z,:,:])
            plt.show()

            # User observes image, and writes down x,y coordinates of rectum 
            coords = self.take_user_input()
            rectum_coords = np.array([coords[0], coords[1], z])

            # Save to csv file
            df = pd.DataFrame({'file_name' : [self.all_file_names[i]], 
                                'x' : [rectum_coords[0]],
                                'y' : [rectum_coords[1]],
                                'z' : [rectum_coords[2]]})
            
            df.to_csv(self.csv_file, index = False, header = False, mode = 'a')

class LabelLesions:
    """
    A class that utilises SITK functions for labelling lesions 
    Output : returns each individual lesion centroid coordinates 
    """

    def __init__(self, origin = (0,0,0), give_centroid_in_mm = False):

        self.origin = origin
        self.give_centroid_in_mm = give_centroid_in_mm #Whether ot not to give centroids in mm or in pixel coords

    def __call__(self, lesion_mask_path):
        """"
        lesion_mask : Image (SITK) sitk.sitkUInt8 type eg sitk.ReadImage(lesion_path, sitk.sitkUInt8)
        """

        # Convert lesion mask from array to image 
        lesion_mask = sitk.ReadImage(lesion_mask_path[0], sitk.sitkUInt8) #uncomment for multipatient_env_v2
        #lesion_mask = sitk.ReadImage(lesion_mask_path, sitk.sitkUInt8)

        #lesion_mask = sitk.GetImageFromArray(lesion_mask_path, sitk.sitkUInt8)
        lesion_mask.SetOrigin(self.origin) 
        #lesion_mask.SetSpacing((0.5, 0.5, 1))

        # Label each lesion within lesion_mask using connected component analysis 
        cc_filter = sitk.ConnectedComponentImageFilter()
        multiple_labels = cc_filter.Execute(lesion_mask) 
        multiple_label_img = sitk.GetArrayFromImage(multiple_labels)
        #print(cc_filter.GetObjectCount())

        # Find centroid of each labelled lesion in mm 
        label_shape_filter= sitk.LabelShapeStatisticsImageFilter()
        #print('Multiple labels treated as a single label and its centroid:')
        label_shape_filter.Execute(multiple_labels)
        lesion_centroids = [label_shape_filter.GetCentroid(i) \
            for i in range(1, label_shape_filter.GetNumberOfLabels()+1)]
        lesion_size = [label_shape_filter.GetPhysicalSize(i) \
            for i in range(1, label_shape_filter.GetNumberOfLabels()+1)]
        lesion_bb = [label_shape_filter.GetBoundingBox(i) \
            for i in range(1, label_shape_filter.GetNumberOfLabels()+1)]
        lesion_perimeter = [label_shape_filter.GetPerimeter(i) \
            for i in range(1, label_shape_filter.GetNumberOfLabels()+1)]
        lesion_diameter = [label_shape_filter.GetEquivalentEllipsoidDiameter(i) \
            for i in range(1, label_shape_filter.GetNumberOfLabels()+1)]

        lesion_statistics = {'lesion_size' : lesion_size, 'lesion_bb' : lesion_bb, 'lesion_diameter' : lesion_diameter} 

        # Convert centroids from mm to pixel coords if not needed in mm 
        if not self.give_centroid_in_mm: 
            lesion_centroids = [lesion_centroid * np.array([2,2,1]) for lesion_centroid in lesion_centroids]
        
        num_lesions = label_shape_filter.GetNumberOfLabels()

        return lesion_centroids, num_lesions, lesion_statistics, np.transpose(multiple_label_img, [1, 2, 0])

class Image_dataloader(Dataset):

    def __init__(self, folder_name, csv_file = '/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv', mode = 'train', use_all = True):
        
        self.folder_name = folder_name
        self.mode = mode
        #self.rectum_df = pd.read_csv(rectum_file)
        #self.all_file_names = self._get_patient_list(os.path.join(self.folder_name, 'lesion'))

        # Obtain list of patient names with multiple lesions -> change to path name
        #df_dataset = pd.read_csv('./patient_data_multiple_lesions.csv')
        #df_dataset = pd.read_csv('/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv')
            # Find which patient indeces have more than 6 lesions 
        df_dataset = pd.read_csv(csv_file)
        patients_w5 = np.where(df_dataset[' num_lesions'] >= 5)[0] # save these indices for next time!!!
    
        # Remove patients where lesions >5 as these are incorrectly labelled!!
        df_dataset = df_dataset.drop(df_dataset.index[patients_w5])
        self.all_file_names = df_dataset['patient_name'].tolist()
        self.num_lesions = df_dataset[' num_lesions'].tolist()
        
        # Write to new csv file 
        #df_dataset.to_csv('/raid/candi/Iani/Biopsy_RL/Updated_patients.csv')
    
        # Train with all patients 
        if use_all:

            size_dataset = len(self.all_file_names)

            train_len = int(size_dataset * 0.7) 
            test_len = int(size_dataset * 0.2) 
            val_len = size_dataset - (train_len + test_len)

            # both test and val have simila rnumber of lesions (mean = 2.4 lesions)
            self.train_names = self.all_file_names[0:train_len]
            self.val_names = self.all_file_names[train_len:train_len + val_len]
            self.test_names = self.all_file_names[train_len + val_len:]
            
            if self.mode == 'train':
                self.all_num_lesions = self.num_lesions[0:train_len]
            elif self.mode == 'val':
                self.all_num_lesions = self.num_lesions[train_len:train_len + val_len]
            else:
                self.all_num_lesions = self.num_lesions[train_len + val_len:]

        # Only train with 105 patients, validate with 15 and validate with 30 : all ahve mean num lesions of 2.6
        else:

            size_dataset = 150 

            train_len = int(size_dataset * 0.7) 
            test_len = int(size_dataset * 0.2) 
            val_len = size_dataset - (train_len + test_len)

            self.train_names = self.all_file_names[0:105]
            self.val_names = self.all_file_names[105:120]
            self.test_names = self.all_file_names[120:150]

            #Defining length of datasets
            #size_dataset = len(self.all_file_names)
 

        self.dataset_len = {'train' : train_len, 'test': test_len, 'val' : val_len}

        # Folder names
        self.lesion_folder = os.path.join(folder_name, 'lesion')
        self.mri_folder = os.path.join(folder_name, 't2w')
        self.prostate_folder = os.path.join(folder_name, 'prostate_mask')

    def _get_patient_list(self, folder_name):
        """
        A function that lists all the patient names
        """
        all_file_names = [f for f in os.listdir(folder_name) if not f.startswith('.')]
        #all_file_paths = [os.path.join(folder_name, file_name) for file_name in self.all_file_names]

        return all_file_names

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

    def __len__(self):
        return self.dataset_len[self.mode]
 
    def __getitem__(self, idx):

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
        lesion_mask = np.transpose((read_img(os.path.join(self.lesion_folder, patient_name))), [1, 2, 0])
        prostate_mask = np.transpose(self._normalise(read_img(os.path.join(self.prostate_folder, patient_name))), [1, 2, 0])
        
        # Get rectum positions
        #rectum_pos = self._get_rectum_pos(patient_name) 
        rectum_pos = self.all_num_lesions[idx] # return as number of lesions 
        sitk_img_path = os.path.join(self.lesion_folder, patient_name)

        return mri_vol, prostate_mask, lesion_mask, sitk_img_path , rectum_pos, patient_name

class Image_dataloader_single(Dataset):

    def __init__(self, folder_name, csv_file = '/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv', idx = 0, mode = 'train'):
        
        self.folder_name = folder_name
        self.mode = mode
        #self.rectum_file = rectum_file 
        #self.rectum_df = pd.read_csv(rectum_file)
        #self.all_file_names = self._get_patient_list(os.path.join(self.folder_name, 'lesion'))
        self.idx = idx
        
        df_dataset = pd.read_csv(csv_file)
        
        #Filter out patients >=5 lesions 
        patients_w5 = np.where(df_dataset[' num_lesions'] >= 5)[0] # save these indices for next time!!!
    
        # Remove patients where lesions >5 as these are incorrectly labelled!!
        df_dataset = df_dataset.drop(df_dataset.index[patients_w5])
        self.all_file_names = df_dataset['patient_name'].tolist()
        self.num_lesions = df_dataset[' num_lesions'].tolist()
        
        #Defining length of datasets
        size_dataset = len(self.all_file_names)

        train_len = int(size_dataset * 0.7) 
        test_len = int(size_dataset * 0.2) 
        val_len = size_dataset - (train_len + test_len)

        self.dataset_len = {'train' : train_len, 'test': test_len, 'val' : val_len}

        # Folder names
        self.lesion_folder = os.path.join(folder_name, 'lesion')
        self.mri_folder = os.path.join(folder_name, 't2w')
        self.prostate_folder = os.path.join(folder_name, 'prostate_mask')

    def _get_patient_list(self, folder_name):
        """
        A function that lists all the patient names
        """
        all_file_names = [f for f in os.listdir(folder_name) if not f.startswith('.')]
        #all_file_paths = [os.path.join(folder_name, file_name) for file_name in self.all_file_names]

        return all_file_names

    def _normalise(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = np.max(img)
        min_img = np.min(img)

        #Normalise values between 0 to 1
        normalised_img =  ((img - min_img)/(max_img - min_img)) 

        return normalised_img.astype(np.float)

    def __len__(self):
        return self.dataset_len[self.mode]
 
    def __getitem__(self, idx):

        if self.mode == 'train':
            idx_ = self.idx

        elif self.mode == 'val':
            idx_ = idx + self.dataset_len['train']
            idx_ = self.idx

        elif self.mode == 'test':
            idx_ = idx + self.dataset_len['train'] + self.dataset_len['val']
            idx_ = self.idx

        # Read prostate mask, lesion mask, prostate mask separately using ImageReader    
        patient_name = self.all_file_names[idx_]
        #patient_name = 'Patient984057610_study_0.nii.gz'
        read_img = ImageReader()
        
        mri_vol = np.transpose(self._normalise(read_img(os.path.join(self.mri_folder, patient_name))), [1, 2, 0])
        lesion_mask = np.transpose(self._normalise(read_img(os.path.join(self.lesion_folder, patient_name))), [1, 2, 0])
        prostate_mask = np.transpose(self._normalise(read_img(os.path.join(self.prostate_folder, patient_name))), [1, 2, 0])
        
        # Get rectum positions
        rectum_pos = self._get_rectum_pos(patient_name) 
        sitk_img_path = os.path.join(self.lesion_folder, patient_name)

        return mri_vol, prostate_mask, lesion_mask, sitk_img_path , rectum_pos, patient_name

class FeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        Correspods to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, multiple_frames = False, num_multiple_frames = 192):
        
        super(FeatureExtractor, self).__init__(observation_space, features_dim)
        # Assumes CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        num_input_channels = observation_space.shape[-1] #rows x cols x channels 
        
        #ResNet - use conv layers only to extract features 
        base_model = torchvision.models.resnet18(pretrained= False)
        self.feature_layers = nn.Sequential(*list(base_model.children())[:-2]) #Obtain feature extraction only

        #Change first layer from 3 input channels to 20     
        if multiple_frames: 
            self.feature_layers[0] = nn.Conv2d(num_multiple_frames, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        #Flatten layers 
        self.flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            all_layers = nn.Sequential(self.feature_layers, self.flatten)
            
            observation_space_shuffled = np.transpose(observation_space.sample(), [2, 1, 0])
            n_flatten = all_layers(torch.as_tensor(observation_space_shuffled[None]).float()).shape[1]
            #n_flatten = all_layers(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]  

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        observations = self._pre_process_image(observations)
        observations = observations.float() 
        output = self.feature_layers(observations.float())
        output = self.flatten(output)
        
        return self.linear(output)

    def _pre_process_image(self, images):
        """ 
        A function that switches the dimension of image from row x col x channel -> channel x row x colmn 
        and addeds a dimension along 0th axis to fit network 
        """

        image = images.clone().detach().to(torch.uint8)
        processed_image = image.permute(0, 3,2,1)
        
        #processed_image = torch.unsqueeze(processed_image, dim= 0)
        return processed_image

class SimpleFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        Correspods to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, multiple_frames = False, num_multiple_frames = 192):
        
        super(SimpleFeatureExtractor, self).__init__(observation_space, features_dim)
        # Assumes CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        num_input_channels = observation_space.shape[-1] #rows x cols x channels 
        
        self.cnn_layers = nn.Sequential(

            # First layer like resnet, stride = 2
            nn.Conv2d(num_multiple_frames, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Apply pooling layer in between 
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        #Flatten layers 
        self.flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            all_layers = nn.Sequential(self.cnn_layers, self.flatten)
            
            observation_space_shuffled = np.transpose(observation_space.sample(), [2, 1, 0])
            n_flatten = all_layers(torch.as_tensor(observation_space_shuffled[None]).float()).shape[1]
            #n_flatten = all_layers(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]  

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        observations = self._pre_process_image(observations)
        observations = observations.float() 
        output = self.cnn_layers(observations.float())
        output = self.flatten(output)
        
        return self.linear(output)

    def _pre_process_image(self, images):
        """ 
        A function that switches the dimension of image from row x col x channel -> channel x row x colmn 
        and addeds a dimension along 0th axis to fit network 
        """
        image = images.clone().detach().to(torch.uint8)
        processed_image = image.permute(0, 3,2,1)
        #processed_image = torch.unsqueeze(processed_image, dim= 0)
        return processed_image

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
        image = images.clone().detach().to(torch.uint8).squeeze()
        if len(image.size()) == 3:
            image = image.unsqueeze(axis=0)
        split_channel_image = torch.cat([torch.cat([image[j,:,:,i*25:(i*25)+25].unsqueeze(0) for i in range(3)]).unsqueeze(0) for j in range(image.size()[0])])#.clone().detach().to(torch.uint8)
        #split_channel_image = torch.cat([torch.cat(torch.tensor_split(image[i,:,:,:].unsqueeze(0), self.num_multiple_frames, dim=3)).unsqueeze(0) for i in range(image.size()[0])])
        #processed_image = image.permute(0, 3,2,1)
        #processed_image = torch.unsqueeze(processed_image, dim= 0)
        
        # Turn image from channel x row x column -> channel x row x column x depth for pre-processing with 3D layers 

        return split_channel_image

class SimpleFeatureExtractor_3D(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        Correspods to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, multiple_frames = False, num_multiple_frames = 3):
        
        super(SimpleFeatureExtractor_3D, self).__init__(observation_space, features_dim)
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
        image = images.clone().detach().to(torch.uint8)
        split_channel_image = torch.cat([torch.cat([image[j,:,:,i*25:(i*25)+25].unsqueeze(0) for i in range(3)]).unsqueeze(0) for j in range(image.size()[0])])#.clone().detach().to(torch.uint8)
        #split_channel_image = torch.cat([torch.cat(torch.tensor_split(image[i,:,:,:].unsqueeze(0), self.num_multiple_frames, dim=3)).unsqueeze(0) for i in range(image.size()[0])])
        #processed_image = image.permute(0, 3,2,1)
        #processed_image = torch.unsqueeze(processed_image, dim= 0)
        
        # Turn image from channel x row x column -> channel x row x column x depth for pre-processing with 3D layers 

        return split_channel_image

class CombinedFeatureExtractor_3D(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        Correspods to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, multiple_frames = False, num_multiple_frames = 192):
        
        super(CombinedFeatureExtractor_3D, self).__init__(observation_space, features_dim)
        # Assumes CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        extractors = {}
        total_feature_size = 0

        # CNN layers for img_volume
        num_multiple_frames = 3
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
            processed_obs_space = self._pre_process_image(torch.as_tensor((observation_space['img_volume'].sample()[None]))).float()
            n_flatten = all_layers(processed_obs_space).shape[1]  

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        # CNN and feature extractors
        extractors = {}
        total_feature_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == 'img_volume':
                extractors[key] = nn.Sequential(self.cnn_layers, self.flatten, self.linear)
                total_feature_size += features_dim
            elif key == 'num_needles_left':
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_feature_size += 16 
        
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_feature_size

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        # Combined features 
        encoded_tensor_list = [] 

        for key, extractor in self.extractors.items(): 
            if key == 'img_volume':
                img_observations = self._pre_process_image(observations[key]).float()
                encoded_tensor_list.append(extractor(img_observations))
            else:
                encoded_tensor_list.append(extractor(observations[key]))

        combined_features = torch.cat(encoded_tensor_list, dim = 1) 

        return combined_features

    def _pre_process_image(self, images):
        """ 
        A function that switches the dimension of image from row x col x channel -> channel x row x colmn 
        and addeds a dimension along 0th axis to fit network 
        """
        image = images.clone().detach().to(torch.uint8)
        #split_channel_image2 = torch.cat([torch.cat([image[j,:,:,i*25:(i*25)+25].unsqueeze(0) for i in range(3)]).unsqueeze(0) for j in range(image.size()[0])])#.clone().detach().to(torch.uint8)
        
        split_channel_image = torch.cat([torch.cat(torch.tensor_split(image[i,:,:,:].unsqueeze(0), 3, dim=3)).unsqueeze(0) for i in range(image.size()[0])])
        #processed_image = image.permute(0, 3,2,1)
        #processed_image = torch.unsqueeze(processed_image, dim= 0)
        
        # Turn image from channel x row x column -> channel x row x column x depth for pre-processing with 3D layers 

        return split_channel_image

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
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

          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.nanmean(y[-self.check_freq:])
              std_reward = np.nanstd(y[-self.check_freq:])

              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} +/- {self.best_mean_reward_std:.2f} \
                - Last mean reward per episode: {mean_reward:.2f} +/- {std_reward:.2f}")

              # Save the model if the mean reward is better than the previously saved mean reward 
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  self.best_mean_reward_std = std_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True

class SaveOnBestTrainingRewardCallback_moreinfo(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback_moreinfo, self).__init__(verbose)
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

def create_grid_coords():
    """
    Creates a grid array of the template grid coordinates

    Notes:
    -------
    Grid array placed 15mm above probe, aligned with probe centre 
    """

    grid_coords_x = np.arange(-30,35, 5)
    grid_coords_y = np.arange(-75, -10, 5)

    x_coords, y_coords = np.meshgrid(grid_coords_x, grid_coords_y)

    return x_coords, y_coords

test_arrays = create_grid_coords()

def create_grid_array(x_idx, y_idx):
    """
    A function that generates grid array coords

    Note: assumes that x_idx, y_idx are in the range (-30,30)

    However, y takes range from (-15,-65) so need to account for this!
    """

    x_grid_pos = x_idx + 35
    y_grid_pos = -y_idx + 35

    grid_array = np.zeros((70,70))

    #Plot a + for where the needle was fired
    grid_array[y_grid_pos, x_grid_pos] = 1
    grid_array[y_grid_pos+1, x_grid_pos] = 1
    grid_array[y_grid_pos-1, x_grid_pos] = 1
    grid_array[y_grid_pos, x_grid_pos-1] = 1
    grid_array[y_grid_pos, x_grid_pos+1] = 1

    return grid_array

#test_grid = create_grid_array(30,-30)
#plt.figure()
#plt.imshow(test_grid)

def obtain_obs(actions):

    # Compute template grid coordinates (x,y) 

    # Find first corresponding lesion positions within template grid (bb)

    # if initialise : initialise within any space on template grid bb of that space 

    # if not initialise : take previous position and add delta_x, delta_y 

    # if trigger : choose next lesion position 
    pass

def find_max_ccl_possible(tumour_mask):
    """
    Finds max ccl possible 

    """

    np.max(tumour_mask ==1, axis = 2)

if __name__ == '__main__':

    lesion_path = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality/lesion/Patient003782816_study_0.nii.gz'
    image_vol = sitk.ReadImage(lesion_path, sitk.sitkUInt8)

    # Testing lesion labeller function 
    #lesion_labeller = LabelLesions(origin = (0,0,0), give_centroid_in_mm = True)
    #lesion_centroids, num_lesions = lesion_labeller(image_vol)
    
    # Testing loading of rectum file 
    rectum_pos_path = '/Users/ianijirahmae/Documents/PhD_project/rectum_pos.csv'
    top_folder = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality'
    img_loader = Image_dataloader(top_folder, rectum_pos_path, mode = 'test')
    Data_sampler = DataSampler(img_loader)

    lesion_labeller = LabelLesions()

    #with open('patient_data_multiple_lesions.csv', 'w') as f:
    #      f.write('''patient_name, num_lesions''')
    
    import csv 
    for idx, data in enumerate(img_loader):
        print(f"idx : {idx} : name : {data[-1]}")
    
    
    for idx, (mri_vol, prostate_mask, tumour_mask, tumour_mask_sitk, rectum_pos, patient_name) in enumerate(img_loader):

        # print patient name in csv file 
        tumour_centroids, num_lesions, tumour_statistics, multiple_label_img = lesion_labeller(tumour_mask_sitk)
        print(f"Patient : {patient_name}, num_lesions : {num_lesions}")

        if num_lesions > 1: 
            writer = csv.writer(open('patient_data_multiple_lesions.csv', 'a'))
            details = [patient_name, str(num_lesions)]
            writer.writerow(details)
        #with open('patient_data.csv', 'a') as f:
        #    details = [patient_name, str(num_lesions)]
            


        
        # if patient_name == 'Patient024798392_study_0.nii.gz':

        #     img_shape = np.shape(tumour_mask)
        #     x_coords = np.arange(0, img_shape[0])
        #     y_coords = np.arange(0, img_shape[1])
        #     z_coords = np.arange(0, img_shape[2])
        #     x,y,z = np.meshgrid(x_coords,y_coords, z_coords)
            
        #     z_mask = z * tumour_mask
        #     z_mask[z_mask == 0] = float("nan")
        #     z_min = np.nanmin(z_mask, axis = 2)
        #     z_max = np.nanmax(z_mask, axis = 2)
        #     max_ccl = np.nanmax(z_max - z_min)

        #     print('chicken')
        
        
        has_no_lesion = (len(np.unique(tumour_mask)) == 1)
        
        if has_no_lesion:
            print(f"Has no lesion : {patient_name}")

        #tumour_centroids, num_lesions, tumour_statistics, multiple_label_img = lesion_labeller(tumour_mask_sitk)
        #prostate_centroid = np.array([100,100,0])
        #tumour_centroids -= prostate_centroid 

        #if num_lesions == 0: 
        #    print(np.shape(tumour_centroids))
        #    print(f"{patient_name} has one lesion only")

        
    print('chicken')


    rectum_df = pd.read_csv(rectum_pos_path)

    # Finding x,y,z of values with file name 
    patient_name = 'Patient482687956_study_0.nii.gz' 
    patient_data = rectum_df[rectum_df['file_name'].str.contains(patient_name)]
    rectum_pos = [patient_data[pos].values[0] for pos in ['x', 'y', 'z']] 

    # Testing dataloader 
    top_folder = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality'
    img_loader = Image_dataloader(top_folder, rectum_pos_path, mode = 'train')

    for i in range(2):
        data = img_loader[i]
        break 

    print('Chicken')

    import pandas as pd
    df_dataset = pd.read_csv('patient_data_multiple_lesions.csv')
    patient_names = df_dataset['patient_name'].tolist()
    num_lesions = df_dataset[' num_lesions'].tolist()

    # Only train with 100 patients, validate with 10 and validate with 20 
    train_names = patient_names[0:100]
    val_names = patient_names[100:110]
    test_names = patient_names[110:130]




    print('chicken')
    # image_vol.SetOrigin((0,0,0)) #set origin to TOP LEFT CORNER -> change for next time 
    # #image_vol.SetSpacing((0.5, 0.5, 1)) changing spacing does not change coordinates of vol as this was found from .nii directly! 

    # # Connected component, then label shape statistics
    # cc_filter = sitk.ConnectedComponentImageFilter()
    # test = cc_filter.Execute(image_vol)
    # #test = sitk.GetArrayFromImage(image_vol)
    # print(cc_filter.GetObjectCount())

    # # Apply multiple lesions 
    # label_shape_filter= sitk.LabelShapeStatisticsImageFilter()
    # label_shape_filter.Execute(test)

    # #background_label = 0
    # print('Multiple labels treated as a single label and its centroid:')
    # label_shape_filter.Execute(test)
    # for label in range(1, label_shape_filter.GetNumberOfLabels()+1):
    #     print(f'label{label}: centroid {label_shape_filter.GetCentroid(label)}')

    # Note centroid labels are from top left corner of MRI volume

    # Note : centroids are given in mm already, not pixel dimensions. If want pixel -> x2 (x,y) coords in mm 


    print('Chicken')
