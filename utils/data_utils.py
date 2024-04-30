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
import copy 
import h5py 

### HELPER FUNCTIONS

def resample_img(img, target_size):
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
    
    
class DiffBiopsyDataset:
    """
    Dataset for sampling biopsy images 
    """
    
    def __init__(self, 
                 dir_name, 
                 mode = 'train'):
        
        self.dir_name = dir_name 
        self.mode = mode # train / test / val 

        # obtain list of names for us and mri labels 
        self.us_names = os.listdir(os.path.join(dir_name, mode, 'us_images'))
        #self.us_label_names = os.listdir(os.path.join(dir_name, mode, 'us_labels'))
        self.mri_names = os.listdir(os.path.join(dir_name, mode, 'mr_images'))
        self.mri_label_names = os.listdir(os.path.join(dir_name, mode, 'mr_labels'))
        self.num_patients = len(self.us_names)
        #self.num_train = num_train # number to train with and validate with after! 

        # Load folder path 
        print(f"Using real images")
 
        self.us_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'us_images', self.us_names[i])).get_fdata().squeeze()) for i in range(self.num_patients)]
        #self.us_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode,'us_labels', self.us_label_names[i])).get_fdata()) for i in range(self.num_data)]

        # Use MR as US for now to avoid error (Erorr raised is ALL_DATA/TESTNEWRLDATA_pix2pix/train/mr_images/case000009.nii.gz is not a gzip file)
        self.mri_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'us_images', self.us_names [i])).get_fdata().squeeze()) for i in range(self.num_patients)] 
        self.mri_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_labels', self.mri_label_names [i])).get_fdata().squeeze()) for i in range(self.num_patients)]
        
        # Check where there are no "targets or lesions" (unsure why this is the case, but doesnt matter, ignore it later)
        check_empty_target = torch.tensor([len(torch.unique(val[:,:,:,1]))==2 for val in self.mri_labels])*1.0
        self.idx_data = np.where(check_empty_target != 0)[0]
        self.num_patients = len(self.idx_data)
        print(f"Num patients with TARGETS : {self.num_patients}")
        
        # Load voxdims of data 
        test_img = (nib.load(os.path.join(dir_name, mode, 'us_images', self.us_names[0])))
        self.voxdims = test_img.header.get_zooms()[1:]
        
        #print(f'chicken')
        # self.mr_path = os.path.join(dir_name, mode, 'mr_images')
        # self.us_path = os.path.join(dir_name, mode, 'us_images')
        # self.labels_path = os.path.join(dir_name, mode, 'mr_labels')
        
    def __getitem__(self, i):
        
        idx = self.idx_data[i] #use idx from those WITH targets 
        
        file_name = self.us_names[idx]
        #print(f" {i} idx :{idx}")
        us = self.us_data[idx] #x 128 x 120 x 128  
        us = us.unsqueeze(0)
        # Add dimesion for "channel"
        #mr_data = torch.tensor(nib.load(os.path.join(self.mr_path, file_name).get_fdata().squeeze())
        #mr_label = torch.tensor(nib.load(os.path.join(self.labels_path, file_name).get_fdata().squeeze())
        #us = torch.tensor(nib.load(os.path.join(self.labels_path, file_name).get_fdata().squeeze())
        
        mr_data = self.mri_data[idx].unsqueeze(0) 
        mr_label = self.mri_labels[idx].unsqueeze(0)

        # normalise data 
        mr = self.normalise_data(mr_data) # 128 x 120 x 128 
        us = self.normalise_data(us)
        mr_label = self.normalise_data(mr_label)

        gland = mr_label[:,:,:,:,0]
        target = mr_label[:,:,:,:,1]
        
        print(f"\n \n \n \n")
        print(f"DEBUGGING : PATIENT NAME: \n")
        print(f"File name : i: {i} idx : {idx} file : {file_name} Target size : {torch.sum(target).item()}")
        print(f"\n \n \n \n")
        # Choose a target with non-zero vals        
        # Checks which targets are non-empty for us label
        # TODO: change dimensions from height width depth to depth widht height for yipeng's code!!!!
        
        # paired : 128 x 128 x 120 size 
        # mr = self.change_order(mr) # change from 128 x 120 x 128 -> 128 x 128 x 120 (dimensions 1 x 128 x 120 x 128 )
        # #us = self.change_order(us.unsqueeze(0))
        # gland = self.change_order(gland)
        # target = self.change_order(target)
        # print(f"idx : {idx} gland size : {gland.size()}")
        target_type = 'Lesion'
        # order : 128 x 120 x 128 
        
        # # for debugging
        # plt.figure()
        # plt.imshow(us.squeeze().cpu()[60,:,:]) # 120 x 128 x 128  : but this way is such that x,y,z and z is axial!!! 
        # plt.savefig("RL_SINGLE_GLAND.png")
        # plt.close()
        
        # Returns in size 1 x 128 x 120 x 128 
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
    
    def compute_lesion_size(self, lesion_files):
        """
        A function to compute lesion size for each file given 
        """

        # obtain lesion files 
        pass 

    def get_lesion_size(self):
        """
        Returns lesion size 
        """
        
        lesion_size = [torch.sum(data[:,:,:,1]).item() for data in self.mri_labels]
        
        return lesion_size 

class SimLesionsDataset:
    """
    Dataset for sampling biopsy images, with simmulated lesions based on other dataset distribution 
    """
    
    def __init__(self, dir_name, h5_path, device, random_seed = 42):
        
        self.dir_name = dir_name 

        # obtain list of names for us and mri labels 
        all_modes = ['train', 'test', 'val']

        ### Load entire dataset for RL generation 
        # Add lists up 
        self.us_data = []   
        self.us_labels = []
        self.mri_data = []
        self.mri_labels = [] 
        
        for mode in all_modes:
            self.us_names = os.listdir(os.path.join(dir_name, mode, 'us_images'))
            self.us_label_names = os.listdir(os.path.join(dir_name, mode, 'us_labels'))
            self.mri_names = os.listdir(os.path.join(dir_name, mode, 'mr_images'))
            self.mri_label_names = os.listdir(os.path.join(dir_name, mode, 'mr_labels'))
            
            num_data = len(self.us_names)
            us_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'us_images', self.us_names[i])).get_fdata().squeeze()) for i in range(num_data)]
            us_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode,'us_labels', self.us_label_names[i])).get_fdata()) for i in range(num_data)]
            mri_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_images', self.mri_names [i])).get_fdata().squeeze()) for i in range(num_data)]
            mri_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_labels', self.mri_label_names [i])).get_fdata().squeeze()) for i in range(num_data)]
    
            self.us_data.extend(us_data)
            self.us_labels.extend(us_labels)
            self.mri_data.extend(mri_data)
            self.mri_labels.extend(mri_labels)
        

        self.num_patients = len(self.us_data)
        print(f"Num patients : {self.num_patients}")
        
        # Load voxdims of data 
        test_img = (nib.load(os.path.join(dir_name, mode, 'us_images', self.us_names[-1])))
        self.voxdims = test_img.header.get_zooms()[1:]
        
        ### Load lesions to sample from 
        h5_file = h5py.File(h5_path, 'r')
        voxdims_all = h5_file['voxdims_all'][()]
        num_patients = len(voxdims_all)
        img_size = (128,120,128) # for down-sampling 
    
        ### Looping through patients!!! ###

        self.all_lesions = [] 
        
        torch.manual_seed(random_seed) # random number generator for obtaining lesions; in order to sample with same lesions for training
        np.random.seed(random_seed) 
        random_patients = np.random.choice(np.arange(300,num_patients), replace = False, size = 205) # randomly sample 200 patients from test set 
        print(random_patients)
        
        for idx in random_patients:
            
            targets = torch.tensor(
                h5_file["/targets_%04d" % idx][()], dtype=torch.uint8, device=device
            )
            
            unique_lesions = torch.unique(targets).cpu()
            num_lesions = len(unique_lesions) -1 
            
            if num_lesions != 0 :
                lesion_id = np.random.choice(unique_lesions[1:]) # not using background idx = 0
            
            else:
                pass 
            
            target_final = ((resample_img(((targets==lesion_id)*1.0).float(),img_size)>0.5)*1.0)#.squeeze()
            
            print(f"Target final unique idxs : {torch.unique(target_final)}")
            intensity_vals = torch.unique(target_final)
            if len(intensity_vals) != 1:
            # Append random subset of lesions to images!
                self.all_lesions.append(target_final)
            else:
                print(f"No lesion found - hence will not add to target dataset")
                
        print(f"Len of dataset : {len(self.all_lesions)}")
        
            # TESTS TO DO: 
            #   1. Lesion is non-empty after sampling!!!
            
            # randomply sample lesiosn 
            
            
            # 2. Check that lesion is NOT outsid eprostate gland ; if ALL OF IT IS OUTSIDE -> CAN'T HELP IT, USE ANYWAY! 
            # i = 0 
            

        
        
        
    def __getitem__(self, i):
        
        #upsample_us = self.resample(self.us_data[idx])
        #upsample_us_labels = self.resample(self.us_labels[idx], label = True)
        
        # TODO : randomly sample lesion labels 
             
        # Obtain idx from self.with_target 
        idx = self.with_target[i]
        
        #print(f" {i} idx :{idx}")
        us = self.us_data[idx]
        us_label = self.us_labels[idx]

        # Add dimesion for "channel"
        mr_data = self.mri_data[idx].unsqueeze(0)
        mr_label = self.mri_labels[idx].unsqueeze(0)
        #mr_label = mr_label[:,:,:,:,0]       # use only prostate label
        
        # normalise data 
        mr = self.normalise_data(mr_data) # 120 x 128 x 128 
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
        
        print(f"idx : {idx} gland size : {gland.size()}")
        
        # TODO: Sample lesions

        ## Code for debugging 
        all_lesion_idxs = list(np.arange(0,len(self.all_lesions)))
        outside_gland = True 
        
        lesion_idx = 0 
        while outside_gland: 

            lesion_idx = (np.random.choice(all_lesion_idxs))
            lesion = self.all_lesions[lesion_idx]
            print(f"Lesion idx : {lesion_idx}")
            
            # us_label = self.normalise_data(self.us_labels[i])
            # gland = 1.0*(us_label[:,:,:,0]>=0.5)
            # gland = self.change_order(gland.unsqueeze(0))
            
            # Assert that lesion is inside gland 
            combined_vol = gland.squeeze() + 2*lesion.squeeze()   
            outside_gland = 2 in torch.unique(combined_vol)    
            
            all_lesion_idxs.remove(lesion_idx)
            
            # TEST to see if all_lesion_idxs are empty
            if len(all_lesion_idxs) == 0:
                print(f"No more lesions to sample from ; sample randomly instead!")
                break 
            
            # TODO : Add registration and deformation to target lesion 
            
            # TODO : unsqueeze to get same dimensions as others!!!
            target = lesion # unsqueeze to get same dimensions as otehrs 
            
            
        # For debugging; 
        # plt.figure()
        # plt.imshow(np.max(combined_vol.numpy()[:,:,:], axis = 0))
        # plt.colorbar()
        # plt.savefig(f"COMBINED_LESION_GLAND_{i}.png")
        # if 2 in torch.unique(combined_vol):
        #     print(f"Lesion outside gland")        


        
        # order : 128 x 120 x 128 
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
 
    def get_lesion_size(self):
        """
        Returns lesion size 
        """
        
        lesion_size = [torch.sum(data[:,:,:,1]).item() for data in self.mri_labels]
        
        return lesion_size 
    
    
class NewBiopsyDataset:
    """
    Dataset for sampling biopsy images 
    """
    
    def __init__(self, dir_name, 
                 give_fake = True,
                 mode = 'train',
                 num_train = 5, 
                 use_both = False,
                 use_all = True):
        
        self.dir_name = dir_name 
        self.mode = mode # train / test / val 
        self.num_train = num_train # number to train with and validate with after! 
        self.use_both = use_both 

        # obtain list of names for us and mri labels 
        
        if mode == 'all':
            all_modes = ['train', 'test', 'val']

            # Add lists up 
            all_us_names = []   
            all_us_label_names = []
            all_mri_names = []
            all_mri_label_names = [] 
            
            for given_mode in all_modes:
                all_us_names.append(os.listdir(os.path.join(dir_name, given_mode, 'us_images')))
                all_us_label_names.append(os.listdir(os.path.join(dir_name, given_mode, 'us_labels')))
                all_mri_names.append(os.listdir(os.path.join(dir_name, given_mode, 'mr_images')))
                all_mri_label_names.append(os.listdir(os.path.join(dir_name, given_mode, 'mr_labels'))) 
                
            # Add up each of the names!!! 
            print('fuecoco')
            complete_list = [] 
            for sublist in all_us_names:
                complete_list.extend(sublist)
            
                                
        else:
            self.us_names = os.listdir(os.path.join(dir_name, mode, 'us_images'))
            self.us_label_names = os.listdir(os.path.join(dir_name, mode, 'us_labels'))
            self.mri_names = os.listdir(os.path.join(dir_name, mode, 'mr_images'))
            self.mri_label_names = os.listdir(os.path.join(dir_name, mode, 'mr_labels'))
            
        
        self.num_data = len(self.us_names)

        # Load folder path 
        
        # Load items 
        if give_fake: 
            print(f"Using fake images")
            folder_us = 'fake_us_images'
        else:
            print(f"Using real images")
            folder_us = 'us_images'
        
        if use_both: 
            # use both real and fake images for training
            self.us_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode,'us_labels', self.us_label_names[i])).get_fdata()) for i in range(self.num_data)]
            self.mri_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_images', self.mri_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
            self.mri_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_labels', self.mri_label_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
            self.us_data_real = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'us_images', self.us_names[i])).get_fdata().squeeze()) for i in range(self.num_data)]
            self.us_data_fake = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'fake_us_images', self.us_names[i])).get_fdata().squeeze()) for i in range(self.num_data)]

        else:    
            self.us_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, folder_us, self.us_names[i])).get_fdata().squeeze()) for i in range(self.num_data)]
            self.us_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode,'us_labels', self.us_label_names[i])).get_fdata()) for i in range(self.num_data)]
            self.mri_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_images', self.mri_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
            self.mri_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_labels', self.mri_label_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        
        self.with_target = np.arange(0,len(self.us_label_names))                
        print(f'chicken')
        # WITH = []
        # WITHOUT = []
        # for i in range(self.num_data):
        #     label = self.mri_labels[i]
        #     with_target = torch.unique(torch.where(label[:,:,:,1:] == 1.0)[-1])
        #     if len(with_target) > 0:
        #         WITH.append(i)
        #     else:
        #         WITHOUT.append(i)
        
        # self.with_target = WITH
        # self.without_target = WITHOUT
        
        
        # If using ALL data
        if use_all:

            # Both : real AND fake 
            if use_both: 
                
                print(f"Using both real and fake dataset")
                
                # Combine REAL AND FAKE DATA sets 
                idx_real = self.with_target[0:self.num_train]
                idx_fake = self.with_target[self.num_train:]
                    
                self.us_data = [] 
                
                for idx in idx_real:
                    self.us_data.append(self.us_data_real[idx])
                
                print(len(self.us_data))
                
                for idx in idx_fake:
                    self.us_data.append(self.us_data_fake[idx])
                
                print(len(self.us_data))
                
                # for data in self.us_data:
                #     print(data.size())
                print(f"Len of us data : {len(self.us_data)} real : {len(idx_real)} fake : {len(idx_fake)}")
            
            # Use real OR fake 
            else:
                print(f"Using fake data {give_fake}")

        # Using subset of data only 
        else:
            
            print(f"Using subset for training only ") 
            # Use a random seed for choosing idx
            np.random.seed(42) 
            all_idxs = np.arange(0,len(self.us_label_names))
            self.with_target = np.random.choice(all_idxs, self.num_train, replace = False)
            print(f"Using subset for training only idxs : {self.with_target}") 
            print('fuecoco')
        
        self.num_patients = len(self.with_target)
        print(f"Num patients : {self.num_patients}")
        
        # Load voxdims of data 
        test_img = (nib.load(os.path.join(dir_name, mode, folder_us, self.us_names[0])))
        self.voxdims = test_img.header.get_zooms()[1:]
        
        print('chicken')
        
    def __getitem__(self, i):
        
        #upsample_us = self.resample(self.us_data[idx])
        #upsample_us_labels = self.resample(self.us_labels[idx], label = True)
                            
        # Obtain idx from self.with_target 
        idx = self.with_target[i]
        
        #print(f" {i} idx :{idx}")
        us = self.us_data[idx]
        us_label = self.us_labels[idx]

        # Add dimesion for "channel"
        mr_data = self.mri_data[idx].unsqueeze(0)
        mr_label = self.mri_labels[idx].unsqueeze(0)
        #mr_label = mr_label[:,:,:,:,0]       # use only prostate label
        
        # normalise data 
        mr = self.normalise_data(mr_data) # 120 x 128 x 128 
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
        
        print(f"idx : {idx} gland size : {gland.size()}")
        
        # order : 128 x 120 x 128 
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
 
    def get_lesion_size(self):
        """
        Returns lesion size 
        """
        
        lesion_size = [torch.sum(data[:,:,:,1]).item() for data in self.mri_labels]
        
        return lesion_size 
    
class BiopsyDataset:
    """
    Dataset for sampling biopsy images 
    """
    
    def __init__(self, dir_name, 
                 mode, 
                 give_fake = False,
                 sub_mode = 'train',
                 use_all = True,
                 num_train = 5, 
                 use_both = False,
                 more_real = False):
        
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
        self.use_both = use_both 

        # Load folder path 
        
        # Load items 
        if give_fake: 
            print(f"Using fake images")
            folder_us = 'fake_us_images'
        else:
            print(f"Using real images")
            folder_us = 'us_images'
        
        if use_both: 
            # use both real and fake images for training
            self.us_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode,'us_labels', self.us_label_names[i])).get_fdata()) for i in range(self.num_data)]
            self.mri_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_images', self.mri_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
            self.mri_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_labels', self.mri_label_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
            self.us_data_real = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'us_images', self.us_names[i])).get_fdata().squeeze()) for i in range(self.num_data)]
            self.us_data_fake = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'fake_us_images', self.us_names[i])).get_fdata().squeeze()) for i in range(self.num_data)]

            test = 'chicken'
        else:    
            
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
        
        # Combine real / fake data for training / validation
        if use_both: 
            
            if more_real: # use more real patients than fake
                idx_real = self.with_target[0:10]
                idx_fake  = self.with_target[10:15]
                idx_val = self.with_target[15:20]
            
            else: # use more fake patients than real 
                idx_real = self.with_target[0:5]
                idx_fake  = self.with_target[5:15]
                idx_val = self.with_target[15:20]
                
            self.us_data = [] 
            
            for idx in idx_real:
                self.us_data.append(self.us_data_real[idx])
            print(len(self.us_data))
            
            for idx in idx_fake:
                self.us_data.append(self.us_data_fake[idx])
            print(len(self.us_data))
            for idx in idx_val:
                self.us_data.append(self.us_data_real[idx])
            print(len(self.us_data))
            
            
            # Repeat for mr labels, us labels, mr data
            mr_labels = copy.deepcopy(self.mri_labels)
            us_labels = copy.deepcopy(self.us_labels)
            mr_data = copy.deepcopy(self.mri_data)

            self.mri_labels = []
            self.us_labels = [] 
            self.mri_data = [] 
            
            for idx in self.with_target:
                self.mri_labels.append(mr_labels[idx])
                self.us_labels.append(us_labels[idx])
                self.mri_data.append(mr_data[idx])

            # for data in self.us_data:
            #     print(data.size())
            print(f"more_real : {more_real} Len of us data : {len(self.us_data)} real : {len(idx_real)} fake : {len(idx_fake)}")
            
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


        # if self.use_both: 
            
        #     if self.more_real:
        #         if (i > 9) and (i < 15):
        #             self.us_data = self.us_data_real
        #         else:
        #             self.us_data = self.us_data_fake
                    
        #     else: # more fake 
        #         if (i > 4) and (i <= 15):
        #             self.us_data = self.us_data_fake
        #         else:
        #             self.us_data = self.us_data_real
        
                            
        if self.mode == 'test':
            
            if self.use_all: 
                
                # if using all, split into 15 train 5 validation 
                if self.sub_mode == 'train':
                    idx_val = np.arange(0,15)[i]
                else:
                    idx_val = np.arange(15,20)[i]
                
                
                # it will always be low
                if self.use_both: 
                    idx = idx_val # juse use main idx 
                else: 
                    idx = self.with_target[idx_val]
                    
                #idx = self.with_target[idx_val]
                
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
        
        print(f"idx : {idx} gland size : {gland.size()}")
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

if __name__ == '__main__':
    
    # For debugging purposes:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    DS_PATH = '/raid/candi/Iani/mr2us/ALL_DATA/trainRL/pix2pix'
    #DS_PATH = '/raid/candi/Iani/mr2us/ALL_DATA/NEWRLDATA_pix2pix'
    #ds = DiffBiopsyDataset(DS_PATH, mode = 'train')
    ds = SimLesionsDataset(DS_PATH)
    test_ds = ds[0]
    
    print('chicken')    
    #dl = DataLoader(ds, batch_size = 1)
    