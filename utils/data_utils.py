import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os 
import numpy as np 
import pandas as pd 
import SimpleITK as sitk
import torch.nn as nn 
# TODO: Convert MR images to US 
import torch 
#import matplotlib
#matplotlib.use('Tkagg')
from matplotlib import pyplot as plt 
import nibabel as nib

class BiopsyDataset:
    """
    Dataset for sampling biopsy images 
    """
    
    def __init__(self, dir_name, 
                 mode, 
                 give_fake = False,
                 sub_mode = 'train',
                 use_all = True,
                 num_train = 5):
        
        self.dir_name = dir_name 
        self.mode = mode 
        self.sub_mode = sub_mode # whehter to use subsect of dataset for training or validation 
        
        # obtain list of names for us and mri labels 
        self.us_names = os.listdir(os.path.join(dir_name, mode, 'us_images'))
        self.us_label_names = os.listdir(os.path.join(dir_name, mode, 'us_labels'))
        self.mri_names = os.listdir(os.path.join(dir_name, mode, 'mr_images'))
        self.mri_label_names = os.listdir(os.path.join(dir_name, mode, 'mr_labels'))
        self.num_data = len(self.us_names)
        self.use_all = use_all 
        self.num_train = num_train # number to train with and validate with after! 
        
        # Load folder path 
        
        # Load items 
        if give_fake: 
            print(f"Using fake images")
            folder_us = 'fake_us_images'
        else:
            print(f"Using real images")
            folder_us = 'us_images'
            
        self.us_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, folder_us, self.us_names[i])).get_fdata().squeeze()) for i in range(self.num_data)]
        self.us_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode,'us_labels', self.us_label_names[i])).get_fdata()) for i in range(self.num_data)]
        self.mri_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_images', self.mri_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        self.mri_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_labels', self.mri_label_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        
        # TODO : find index of where there are targets available!!!
        WITH = []
        WITHOUT = []
        for i in range(self.num_data):
            label = self.mri_labels[i]
            with_target = torch.unique(torch.where(label[:,:,:,1:] == 1.0)[-1])
            if len(with_target) > 0:
                WITH.append(i)
            else:
                WITHOUT.append(i)
        
        self.with_target = WITH
        self.without_target = WITHOUT
        
        if self.mode == 'test':
            
            if self.use_all: 
                if self.sub_mode == 'train':
                    self.num_patients = 15 # use 15 for training 
                else:
                    self.num_patients = 5 # 5 for validating 
            
            # using subset for training 
            else:
                if self.sub_mode == 'train':
                    self.num_patients = self.num_train # use n number for training! 
                    
                else:
                    self.num_patients = 5 # 5 for validating 
                    
        # no submode for test patients
        
        else:
            self.num_patients = len(self.with_target)
        
        # Load voxdims of data 
        test_img = (nib.load(os.path.join(dir_name, mode, folder_us, self.us_names[0])))
        self.voxdims = test_img.header.get_zooms()[1:]
        
        print('chicken')
        
    def __getitem__(self, i):
        
        #upsample_us = self.resample(self.us_data[idx])
        #upsample_us_labels = self.resample(self.us_labels[idx], label = True)

        if self.mode == 'test':
            
            if self.use_all: 
                # if using all, split into 15 train 5 validation 
                if self.sub_mode == 'train':
                    idx_val = np.arange(0,15)[i]
                else:
                    idx_val = np.arange(15,20)[i]
                
                idx = self.with_target[idx_val]
                
            else:
                # if using some, split into 5 train, 5 validation 
                
                if self.sub_mode == 'train':
                    idx_val = np.arange(0,self.num_train)[i] # use first N number of patients to train with 
                else:
                    idx_val = np.arange(15,20)[i] # use last 5
                
                idx = self.with_target[idx_val]
            
        else:
            
            idx = self.with_target[i]
        
        #print(f" {i} idx :{idx}")
        # no need to transpose if alligned alerady
        us = self.us_data[idx]
        us_label = self.us_labels[idx]

        # Add dimesion for "channel"
        mr_data = self.mri_data[idx].unsqueeze(0)
        mr_label = self.mri_labels[idx].unsqueeze(0)
        #mr_label = mr_label[:,:,:,:,0]       # use only prostate label
        
        # normalise data 
        mr = self.normalise_data(mr_data)
        us = self.normalise_data(us)
        mr_label = self.normalise_data(mr_label)
        us_label = self.normalise_data(us_label)

        gland = 1.0*(us_label[:,:,:,0]>=0.5)
        
        # Choose a target with non-zero vals        
        # Checks which targets are non-empty for us label
        t_idx = torch.unique(torch.where((mr_label[:,:,:,:,1:]) == 1)[-1])
        
        # Return lesion mask from MR registered!!!
        
        if len(t_idx) == 0:
            target = mr_label[:,:,:,:,1].squeeze()
            target_type = 'None'
        else:
            # Randomly sample target / ROI
            if 0 in t_idx: # ie lesion available, use this 
                # add t_idx+1 because 1st index is prostate gland!!
                target = mr_label[:,:,:,:,t_idx[0]+1].squeeze()
                target_type = 'Lesion'
            # if no lesion available, use calcification / 
            else:
                roi_idx = np.random.choice(t_idx)
                target = mr_label[:,:,:,:,roi_idx+1].squeeze()
                target_type = 'Other'
            
            # TODO: change dimensions from height width depth to depth widht height for yipeng's code!!!!
            mr = self.change_order(mr)
            us = self.change_order(us.unsqueeze(0))
            gland = self.change_order(gland.unsqueeze(0))
            target = 1.0*(self.change_order(target.unsqueeze(0)) >= 0.5)
            
        return mr, us, gland, target, target_type, self.voxdims
    
    def __len__(self):
        return self.num_patients
    
    def normalise_data(self, img):
        """
        Normalises labels and images 
        """
        
        min_val = torch.min(img)
        max_val = torch.max(img)
        
        if max_val == 0: 
            #print(f"Empty mask, not normalised img")
            norm_img = img # return as 0s only if blank image or volume 
        else: 
            norm_img = (img - min_val) / (max_val - min_val)
        
        return norm_img 
    
    def change_order(self, tensor_data):
        """
        Changes order of tensor from width height depth to edpth width height 
        """
        
        return tensor_data.permute(0,3,1,2)
 
 
class MR2US:
    """
    Class which implements conversion from MR2US to obtain US slices
    """
    def __init__(self, model, img_size = torch.tensor([120,128,128])):
        
        self.model = model 
        self.img_size = img_size
        
    def convert_mr2us(self, mr_dataloader):
        """
        Function which converts MR2US given MR dataset
        """

        us_vols = []
        
        for i, (mr, prostate_mask, lesion_mask, sitk_img_path , rectum_pos, patient_name) in enumerate(mr_dataloader):
            
            mr_size = mr.size()[1:]
            
            if mr_size != self.img_size:
                
                # Resample first 
                mr = self.resample_img(mr, self.img_size)
            
            # Obtain US images
            us = self.model(mr)

            # Re-sample to original size
            us_resampled = self.resample_img(us, mr_size)
            us_vols.append(us)
            
            # sanity check 
            num_slices = us_resampled.size()[-1]
            
            fig, axs = plt.subplots(1,3)
            for i in range(0,num_slices, 10):
                us_ax = us_resampled.squeeze().detach().numpy()[:,:,i]
                us_sag = us_resampled.squeeze().detach().numpy()[:,50,:]
                mr_img = mr.squeeze().detach().numpy()[:,:,i]
                axs[0].imshow(us_ax)
                axs[1].imshow(mr_img)
                axs[2].imshow(us_sag)
                plt.pause(0.1)
            
            print('fuecoco')
        return us_vols 
    
    def convert_img(self, mr):
        """
        Function for converting mr2us per image, instead of giving whole dataloader 
        """
        total_size = mr.size()
        if len(total_size) == 3:
            mr_size = total_size
        else:
            mr_size = total_size[1:]
            
        if mr_size != self.img_size:
            
            # Resample first to suit dimensions of trained model 
            mr = self.resample_img(mr, self.img_size)
        
        # Obtain US images
        us = self.model(mr)

        # Re-sample to original size
        us_resampled = self.resample_img(us, mr_size)
        
        return us_resampled 
    
    def resample_img(self, img, target_size):
        """
        Resample a 3D tensor based on the target size using PyTorch.

        Parameters:
        - tensor: 3D PyTorch tensor representing the input tensor.
        - target_size: Tuple of three integers (target_depth, target_height, target_width).

        Returns:
        - Resampled tensor with the specified size. : batch_size x channel x height x width x depth 
        """

        # Use PyTorch's interpolation function (trilinear interpolation)
        if len(img.size()) == 4: # for resampling mr 
            resampled_tensor = F.interpolate(img.unsqueeze(0), size=tuple(target_size), mode='trilinear', align_corners=False)
        elif len(img.size()) == 3:
             resampled_tensor = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=tuple(target_size), mode='trilinear', align_corners=False)
        else: 
            resampled_tensor = F.interpolate(img, size=tuple(target_size), mode='trilinear', align_corners=False)
        #resampled_tensor = resampled_tensor.squeeze(0).squeeze(0)

        return resampled_tensor

class ImageReader:

    def __call__(self, file_path, require_sitk_img = False):
        image_vol = sitk.ReadImage(file_path)
        #voxdims = image_vol.GetSpacing()[::-1]
        #print(f"Voxdims : {voxdims}")
        image_vol = sitk.GetArrayFromImage(image_vol)
        image_size = np.shape(image_vol)

        if require_sitk_img: 
            sitk_image_vol = sitk.ReadImage(file_path, sitk.sitkUInt8)
            voxdims = sitk_image_vol.GetSpacing()[::-1]
            print(f"Voxdims : {voxdims}")
            return image_vol, sitk_image_vol

        else:        
            return image_vol
        
class MR_dataset(Dataset):

    def __init__(self, folder_name, csv_file = '/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv', mode = 'train', use_all = True):
        
        self.folder_name = folder_name
        self.mode = mode
        
        # Find which patient indeces have more than 6 lesions 
        df_dataset = pd.read_csv(csv_file)
        patients_w5 = np.where(df_dataset[' num_lesions'] >= 5)[0] # save these indices for next time!!!
    
        # Remove patients where lesions >5 as these are incorrectly labelled!!
        df_dataset = df_dataset.drop(df_dataset.index[patients_w5])
        self.all_file_names = df_dataset['patient_name'].tolist()
        self.num_lesions = df_dataset[' num_lesions'].tolist()

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
        rectum_pos = 0 
        sitk_img_path = os.path.join(self.lesion_folder, patient_name)

        return mri_vol, prostate_mask, lesion_mask, sitk_img_path , rectum_pos, patient_name
