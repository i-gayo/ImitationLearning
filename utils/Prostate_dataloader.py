#Pytorch DataLoader samples 
from torch.utils.data import Dataset, DataLoader, RandomSampler 
import numpy as np 
import os 
import h5py
import torch 
import copy

class ProstateDataset(Dataset):
    def __init__(self, folder_name, file_rectum, mode = 'train', normalise = False):
    
        self.folder_name = folder_name
        self.file_rectum = file_rectum 
        self.mode = mode
        self.rectum_position = np.genfromtxt(self.file_rectum, delimiter = ',', skip_header = 1, usecols = (1,2,3))
        self.normalise = normalise
        
        #Defining length of datasets
        self.train_len = 38 #70% 
        #self.val_len = 5 #10%
        self.test_len = 15 #30%

    def __len__(self):
        
        if self.mode == 'train':
            return self.train_len

        #Holdout set 
        elif self.mode == 'test':
            return self.test_len
        
    def __getitem__(self, idx):

        if self.mode == 'train':
            idx_ = idx
            #print(f"Training idx {idx_}")
            file_name = 'PS_data_' + str(idx_) + '.h5'

        #elif self.mode == 'val':
        #    idx_ = idx + self.train_len
        #    file_name = 'PS_data_' + str(idx_) + '.h5'

        elif self.mode == 'test':
            idx_ = idx+self.train_len
            #print(f"Testing idx {idx_}")
            file_name = 'PS_data_' + str(idx_) + '.h5'

        #"PS_data_%.h5" % idx
        dataset = self._load_h5_file(file_name)
        #print(file_name)

        #Extracting volume datasets: need to turn into torch objetcts: np.asarray
        prostate_mask = np.array(dataset['prostate_mask'])
        tumour_mask = np.array(dataset['tumour_mask'])
        mri_vol = np.array(dataset['mri_vol'])
        rectum_pos = self.rectum_position[idx_]

        #Normalise dataset between 0-255
        if self.normalise: 
            prostate_n = self._convert_to_uint8(prostate_mask)
            tumour_n = self._convert_to_uint8(tumour_mask)
            mri_n = self._convert_to_uint8(mri_vol)

            prostate_mask = copy.deepcopy(prostate_n)
            tumour_mask = copy.deepcopy(tumour_n)
            mri_vol = copy.deepcopy(mri_n)

        return mri_vol, prostate_mask, tumour_mask, rectum_pos

    def _load_h5_file(self, filename):
        filename = os.path.join(self.folder_name, filename)
        self.h5_file = h5py.File(filename, 'r')
        return self.h5_file

    def _convert_to_uint8(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = img.max()
        min_img = img.min()

        #Normalise values between 0 to 1
        normalised_img = 255* (img - min_img)/(max_img - min_img)

        return normalised_img.astype(np.uint8)

class ProstateDataset_combine_testval(Dataset):
    
    def __init__(self, folder_name, file_rectum, mode = 'train', normalise = True, augment = False):
    
        self.folder_name = folder_name
        self.file_rectum = file_rectum 
        self.mode = mode
        self.rectum_position = np.genfromtxt(self.file_rectum, delimiter = ',', skip_header = 1, usecols = (1,2,3))
        self.normalise = normalise
        self.augment = augment  

        #Defining length of datasets
        self.train_len = 38 #70% 
        self.val_len = 6 #10%
        self.test_len = 15 #30%

    def __len__(self):
        
        if self.mode == 'train':
            return self.train_len

        if self.mode == 'val':
            return self.val_len

        #Holdout set 
        elif self.mode == 'test':
            return self.test_len
        
    def __getitem__(self, idx):

        if self.mode == 'train':
            idx_ = idx
            #print(f"Training idx {idx_}")
            file_name = 'PS_data_' + str(idx_) + '.h5'

        elif self.mode == 'val':
            idx_ = idx + self.train_len
            file_name = 'PS_data_' + str(idx_) + '.h5'

        elif self.mode == 'test':
            idx_ = idx + self.train_len + self.val_len
            #print(f"Testing idx {idx_}")
            file_name = 'PS_data_' + str(idx_) + '.h5'

        #"PS_data_%.h5" % idx
        dataset = self._load_h5_file(file_name)
        #print(file_name)

        #Extracting volume datasets: need to turn into torch objetcts: np.asarray
        mri_vol = np.array(dataset['mri_vol'])
        prostate_mask = np.array(dataset['prostate_mask'])
        tumour_mask = np.array(dataset['tumour_mask'])
        rectum_pos = self.rectum_position[idx_]

        #Normalise dataset between 0-255
        if self.normalise: 
            prostate_n = self._normalise(prostate_mask)
            tumour_n = self._normalise(tumour_mask)
            mri_n = self._normalise(mri_vol)

            prostate_mask = copy.deepcopy(prostate_n)
            tumour_mask = copy.deepcopy(tumour_n)
            mri_vol = copy.deepcopy(mri_n)

        if self.augment: 
            
            #Only augment for 30% of the time 
            prob = np.random.rand()
            if prob < 0.3: 
                flipped_images, rectum_a = self._augment([mri_vol, prostate_mask, tumour_mask], rectum_pos)

                mri_vol = copy.deepcopy(flipped_images[0])
                prostate_mask = copy.deepcopy(flipped_images[1])
                tumour_mask = copy.deepcopy(flipped_images[2])
                rectum_pos = copy.deepcopy(rectum_a)

        return mri_vol, prostate_mask, tumour_mask, rectum_pos

    def _load_h5_file(self, filename):
        filename = os.path.join(self.folder_name, filename)
        self.h5_file = h5py.File(filename, 'r')
        return self.h5_file

    def _convert_to_uint8(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = img.max()
        min_img = img.min()

        #Normalise values between 0 to 1
        normalised_img = 255* (img - min_img)/(max_img - min_img)

        return normalised_img.astype(np.uint8)
    
    def _normalise(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = img.max()
        min_img = img.min()

        #Normalise values between 0 to 1
        normalised_img =  ((img - min_img)/(max_img - min_img)) 

        return normalised_img.astype(np.float)

    def _augment(self, images, rectum_pos):

        flipped_images = [] 
        for img in images: 
            flipped_img = np.flip(img, axis = 1)
            flipped_images.append(flipped_img)
        
        #Flip rectum pos too 
        flipped_rectum_pos = copy.deepcopy(rectum_pos)
        flipped_rectum_pos[0] = 160 - rectum_pos[0]

        return flipped_images, flipped_rectum_pos

class ProstateDataset_normalised(Dataset):
    
    def __init__(self, folder_name, file_rectum, mode = 'train', normalise = True, augment = False):
    
        self.folder_name = folder_name
        self.file_rectum = file_rectum 
        self.mode = mode
        self.rectum_position = np.genfromtxt(self.file_rectum, delimiter = ',', skip_header = 1, usecols = (1,2,3))
        self.normalise = normalise
        self.augment = augment  

        #Defining length of datasets
        self.train_len = 38 #70% 
        self.val_len = 6 #10%
        self.test_len = 9 #30%

    def __len__(self):
        
        if self.mode == 'train':
            return self.train_len

        if self.mode == 'val':
            return self.val_len

        #Holdout set 
        elif self.mode == 'test':
            return self.test_len
        
    def __getitem__(self, idx):

        if self.mode == 'train':
            idx_ = idx
            #print(f"Training idx {idx_}")
            file_name = 'PS_data_' + str(idx_) + '.h5'

        elif self.mode == 'val':
            idx_ = idx + self.train_len
            print(f"Testing idx {idx_}")
            file_name = 'PS_data_' + str(idx_) + '.h5'

        elif self.mode == 'test':
            idx_ = idx + self.train_len + self.val_len
            print(f"Testing idx {idx_}")
            file_name = 'PS_data_' + str(idx_) + '.h5'

        #"PS_data_%.h5" % idx
        dataset = self._load_h5_file(file_name)
        #print(file_name)

        #Extracting volume datasets: need to turn into torch objetcts: np.asarray
        mri_vol = np.array(dataset['mri_vol'])
        prostate_mask = np.array(dataset['prostate_mask'])
        tumour_mask = np.array(dataset['tumour_mask'])
        rectum_pos = self.rectum_position[idx_]

        #Normalise dataset between 0-255
        if self.normalise: 
            prostate_n = self._normalise(prostate_mask)
            tumour_n = self._normalise(tumour_mask)
            mri_n = self._normalise(mri_vol)

            prostate_mask = copy.deepcopy(prostate_n)
            tumour_mask = copy.deepcopy(tumour_n)
            mri_vol = copy.deepcopy(mri_n)

        if self.augment: 
            
            #Only augment for 30% of the time 
            prob = np.random.rand()
            if prob < 0.3: 
                flipped_images, rectum_a = self._augment([mri_vol, prostate_mask, tumour_mask], rectum_pos)

                mri_vol = copy.deepcopy(flipped_images[0])
                prostate_mask = copy.deepcopy(flipped_images[1])
                tumour_mask = copy.deepcopy(flipped_images[2])
                rectum_pos = copy.deepcopy(rectum_a)

        return mri_vol, prostate_mask, tumour_mask, rectum_pos

    def _load_h5_file(self, filename):
        filename = os.path.join(self.folder_name, filename)
        self.h5_file = h5py.File(filename, 'r')
        return self.h5_file

    def _convert_to_uint8(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = img.max()
        min_img = img.min()

        #Normalise values between 0 to 1
        normalised_img = 255* (img - min_img)/(max_img - min_img)

        return normalised_img.astype(np.uint8)
    
    def _normalise(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = img.max()
        min_img = img.min()

        #Normalise values between 0 to 1
        normalised_img =  ((img - min_img)/(max_img - min_img)) 

        return normalised_img.astype(np.float)

    def _augment(self, images, rectum_pos):

        flipped_images = [] 
        for img in images: 
            flipped_img = np.flip(img, axis = 1)
            flipped_images.append(flipped_img)
        
        #Flip rectum pos too 
        flipped_rectum_pos = copy.deepcopy(rectum_pos)
        flipped_rectum_pos[0] = 160 - rectum_pos[0]

        return flipped_images, flipped_rectum_pos

class ProstateDataset_two(Dataset):
    
    def __init__(self, folder_name, file_rectum, indexes = [1,3], mode = 'train', normalise = True, augment = False):
    
        self.folder_name = folder_name
        self.file_rectum = file_rectum 
        self.mode = mode
        self.rectum_position = np.genfromtxt(self.file_rectum, delimiter = ',', skip_header = 1, usecols = (1,2,3))
        self.normalise = normalise
        self.augment = augment  
        self.indexes = indexes #Index of datasets to use for training 

        #Defining length of datasets
        self.train_len = 2 #70% 
        self.val_len = 6 #10%
        self.test_len = 9 #30%

    def __len__(self):
        
        if self.mode == 'train':
            return self.train_len

        if self.mode == 'val':
            return self.val_len

        #Holdout set 
        elif self.mode == 'test':
            return self.test_len
        
    def __getitem__(self, idx):

        if self.mode == 'train':
            idx_ = self.indexes[idx]
            #print(f"Training idx {idx_}")
            file_name = 'PS_data_' + str(idx_) + '.h5'

        elif self.mode == 'val':
            idx_ = idx + self.train_len
            #print(f"Testing idx {idx_}")
            file_name = 'PS_data_' + str(idx_) + '.h5'

        elif self.mode == 'test':
            idx_ = idx + self.train_len + self.val_len
            print(f"Testing idx {idx_}")
            file_name = 'PS_data_' + str(idx_) + '.h5'

        #"PS_data_%.h5" % idx
        dataset = self._load_h5_file(file_name)
        #print(file_name)

        #Extracting volume datasets: need to turn into torch objetcts: np.asarray
        mri_vol = np.array(dataset['mri_vol'])
        prostate_mask = np.array(dataset['prostate_mask'])
        tumour_mask = np.array(dataset['tumour_mask'])
        rectum_pos = self.rectum_position[idx_]

        #Normalise dataset between 0-255
        if self.normalise: 
            prostate_n = self._normalise(prostate_mask)
            tumour_n = self._normalise(tumour_mask)
            mri_n = self._normalise(mri_vol)

            prostate_mask = copy.deepcopy(prostate_n)
            tumour_mask = copy.deepcopy(tumour_n)
            mri_vol = copy.deepcopy(mri_n)

        if self.augment: 
            
            #Only augment for 30% of the time 
            prob = np.random.rand()
            if prob < 0.3: 
                flipped_images, rectum_a = self._augment([mri_vol, prostate_mask, tumour_mask], rectum_pos)

                mri_vol = copy.deepcopy(flipped_images[0])
                prostate_mask = copy.deepcopy(flipped_images[1])
                tumour_mask = copy.deepcopy(flipped_images[2])
                rectum_pos = copy.deepcopy(rectum_a)

        return mri_vol, prostate_mask, tumour_mask, rectum_pos

    def _load_h5_file(self, filename):
        filename = os.path.join(self.folder_name, filename)
        self.h5_file = h5py.File(filename, 'r')
        return self.h5_file

    def _convert_to_uint8(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = img.max()
        min_img = img.min()

        #Normalise values between 0 to 1
        normalised_img = 255* (img - min_img)/(max_img - min_img)

        return normalised_img.astype(np.uint8)
    
    def _normalise(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = img.max()
        min_img = img.min()

        #Normalise values between 0 to 1
        normalised_img =  ((img - min_img)/(max_img - min_img)) 

        return normalised_img.astype(np.float)

    def _augment(self, images, rectum_pos):

        flipped_images = [] 
        for img in images: 
            flipped_img = np.flip(img, axis = 1)
            flipped_images.append(flipped_img)
        
        #Flip rectum pos too 
        flipped_rectum_pos = copy.deepcopy(rectum_pos)
        flipped_rectum_pos[0] = 160 - rectum_pos[0]

        return flipped_images, flipped_rectum_pos

class ProstateDataset_normalised_single(Dataset):
    def __init__(self, folder_name, file_rectum, idx = 0, mode = 'train', normalise = True, augment = False):
    
        self.folder_name = folder_name
        self.file_rectum = file_rectum 
        self.idx = idx 
        self.mode = mode
        self.rectum_position = np.genfromtxt(self.file_rectum, delimiter = ',', skip_header = 1, usecols = (1,2,3))
        self.normalise = normalise
        self.augment = augment  

        #Defining length of datasets
        self.train_len = 38 #70% 
        self.val_len = 6 #10%
        self.test_len = 9 #30%

    def __len__(self):
        
        if self.mode == 'train':
            return self.train_len

        if self.mode == 'val':
            return self.val_len

        #Holdout set 
        elif self.mode == 'test':
            return self.test_len
        
    def __getitem__(self, idx):

        if self.mode == 'train':
            idx_ = self.idx #Previously 29 only 
            #print(f"Training idx {idx_}")
            file_name = 'PS_data_' + str(idx_) + '.h5'

        elif self.mode == 'val':
            idx_ = idx + self.train_len
            file_name = 'PS_data_' + str(idx_) + '.h5'

        elif self.mode == 'test':
            idx_ = idx + self.train_len + self.val_len
            #print(f"Testing idx {idx_}")
            file_name = 'PS_data_' + str(idx_) + '.h5'

        #"PS_data_%.h5" % idx
        dataset = self._load_h5_file(file_name)
        #print(file_name)

        #Extracting volume datasets: need to turn into torch objetcts: np.asarray
        mri_vol = np.array(dataset['mri_vol'])
        prostate_mask = np.array(dataset['prostate_mask'])
        tumour_mask = np.array(dataset['tumour_mask'])
        rectum_pos = self.rectum_position[idx_]

        #Normalise dataset between 0-255
        if self.normalise: 
            prostate_n = self._normalise(prostate_mask)
            tumour_n = self._normalise(tumour_mask)
            mri_n = self._normalise(mri_vol)

            prostate_mask = copy.deepcopy(prostate_n)
            tumour_mask = copy.deepcopy(tumour_n)
            mri_vol = copy.deepcopy(mri_n)

        if self.augment: 
            
            #Only augment for 30% of the time 
            prob = np.random.rand()
            if prob < 0.5: 
                flipped_images, rectum_a = self._augment([mri_vol, prostate_mask, tumour_mask], rectum_pos)

                mri_vol = copy.deepcopy(flipped_images[0])
                prostate_mask = copy.deepcopy(flipped_images[1])
                tumour_mask = copy.deepcopy(flipped_images[2])
                rectum_pos = copy.deepcopy(rectum_a)

        return mri_vol, prostate_mask, tumour_mask, rectum_pos

    def _load_h5_file(self, filename):
        filename = os.path.join(self.folder_name, filename)
        self.h5_file = h5py.File(filename, 'r')
        return self.h5_file

    def _convert_to_uint8(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = img.max()
        min_img = img.min()

        #Normalise values between 0 to 1
        normalised_img = 255* (img - min_img)/(max_img - min_img)

        return normalised_img.astype(np.uint8)
    
    def _normalise(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = img.max()
        min_img = img.min()

        #Normalise values between 0 to 1
        normalised_img =  ((img - min_img)/(max_img - min_img)) 

        return normalised_img.astype(np.float)

    def _augment(self, images, rectum_pos):

        flipped_images = [] 
        for img in images: 
            flipped_img = np.flip(img, axis = 1)
            flipped_images.append(flipped_img)
        
        #Flip rectum pos too 
        flipped_rectum_pos = copy.deepcopy(rectum_pos)
        flipped_rectum_pos[0] = 160 - rectum_pos[0]

        return flipped_images, flipped_rectum_pos

class ImitationLearningDataset(Dataset):
    """
    Obtains ground truth patient specific data for training 
    """

    def __init__(self, patient_idx, mode = 'train', shuffled_idx = None):
        """
        Parameters:
        ------------
        :patient_idx: Patient index to obtain data for
        :mode: (str) Which dataset to obtain, training, val or test
        :shuffled_idx: list of np arrays ([len_34, len_5, len_10]) of 
        shuffled idx to use for train, test and validation 
        """
        self.idx_map = {'train': 0, 'val': 1, 'test': 2}
        self.patient_idx = patient_idx 
        self.mode = mode 

        if shuffled_idx == None: 
            self.dataset_idx = [np.arange(0,34), np.arange(34, 39), np.arange(39, 49)]
        else:
            self.dataset_idx = shuffled_idx 

        file_name = 'patient' + str(patient_idx) + '_il.h5'

        #Open and save file contents 
        self.data = h5py.File(file_name, 'r')

        #Go over each positions, observations and stack together for all dataset_idx in this training, testing or validating 
        self.all_obs, self.all_actions, self.dataset_size = self._stack_dataset()
            
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        
        action = self.all_actions[idx,:] 
        obs = self.all_obs[idx, :,:,:]

        return action, obs 
    
    def _stack_dataset(self):
        """
        A function that stacks datasets on top of each other

        Returns:
        -------
        :dataset_size:
        
        """

        #Obtain idx vals for train, val or test 
        idx_map = self.idx_map[self.mode]

        all_obs = [] 
        all_actions = [] 

        for idx in self.dataset_idx[idx_map]:
            
            #Load items
            data_group_name = 'pos_' + str(int(idx))
            pos_data = self.data.get(data_group_name)
            obs = np.array(pos_data.get('obs'))
            actions = np.array(pos_data.get('actions'))
            
            #Stack each corresopnding actions and observations into list 
            all_obs.append(obs)
            all_actions.append(actions)

        #Stack obs and actions to single dataset for training 
        all_obs = torch.from_numpy(np.concatenate(all_obs))
        all_actions = torch.from_numpy(np.concatenate(all_actions))

        dataset_size = all_actions.size()[0]

        return all_obs, all_actions, dataset_size 

    def _alternative_getitem(self, idx):
            #Obtain corresponding dataidx 
        if self.mode == 'train':
            data_idx = self.dataset_idx[0][idx]
        elif self.mode == 'val':
            data_idx = self.dataset_idx[1][idx]
        else:
            data_idx = self.dataset_idx[2][idx]
    
        data_group_name = 'pos_' + str(int(data_idx))
        pos_data = self.data.get(data_group_name)
        #pos_1_raw_actions = np.array(pos_data.get('raw_actions'))
        #pos_1_current_pos = np.array(pos_1.get('grid_pos'))

        obs = np.array(pos_data.get('obs'))
        actions = np.array(pos_data.get('actions'))

        #Turn into torch arrays
        obs = torch.from_numpy(obs)
        actions = torch.from_numpy(actions)

        return torch.unsqueeze(obs, dim = 0), torch.unsqueeze(actions, dim =0)


    
class DataSampler(ProstateDataset):
    """
    DataSampler class that deals with sampling data from training, testing validation 
    
    Consists of a Dataset and DataLoader class 

    """
    def __init__(self, ProstateDataset):
        
        
        self.PS_dataset = ProstateDataset

        self.PS_Dataloader = DataLoader(self.PS_dataset, batch_size = 1, shuffle =  False)
        self.iterator = iter(self.PS_Dataloader)
        
        #Initialise internal counter that checks how many times a data has been sampled
        self.data_counter = 0 
        self.data_size = len(self.PS_dataset)

    def sample_data(self):
        """
        Samples next data using PS_iter
        """
        
        try:
            data = next(self.iterator)
        
        #If stopiteration is raised, re-start the iterator 
        except StopIteration:
            self._restart_iteration()
            data = next(self.iterator)
        
        #Update data counter
        self.data_counter += 1

        return data
    
    def _restart_iteration(self):

        #Restart iteration 
        #self.PS_Dataloader = DataLoader(self.PS_dataset, batch_size = 1, shuffle =  False)
        self.iterator = iter(self.PS_Dataloader) 
        self.data_counter == 0 

#Training Dataset 
#PS_dataset = ProstateDataset_normalised('../Prostate_dataset', '/Users/iani/Documents/MRes/Project/Code_GPU/rectum_labels.csv',mode= 'train', augment = True)
#PS_dataloader = DataLoader(PS_dataset, batch_size = 1, shuffle =  True) #, sampler = RandomSampler) 

#for idx_, (mri_vol, tumour_mask, prostate_mask, rectum_pos) in enumerate(PS_dataloader):
#    #print(np.shape((rectum_pos.numpy())))
#    print(np.shape(mri_vol))
    #break
   
"""
class ProstateDataset(torch.utils.data.Dataset):
    def __init__(self, folder_name, file_rectum, is_train=True):
        self.folder_name = folder_name
        self.file_rectum = file_rectum 
        self.is_train = is_train
        self.rectum_position = np.genfromtxt(self.file_rectum, delimiter = ',', skip_header = 1, usecols = (1,2,3))

    def __len__(self):
        return (50 if self.is_train else 30)

    def __getitem__(self, idx):
        if self.is_train:
            file_name = 'PS_data_' + str(idx) + '.h5'
            dataset = self._load_h5_file("PS_data_%.h5" % idx)

            #Extracting volume datasets: need to turn into torch objetcts: np.asarray
            prostate_mask = np.array(dataset['prostate_mask'])
            tumour_mask = np.array(dataset['tumour_mask'])
            mri_vol = np.array(dataset['mri_vol'])
            rectum_pos = self.rectum_position[idx]
            return mri_vol, prostate_mask, tumour_mask, rectum_pos 
        else:
            pass

    def _load_h5_file(self, filename):
        filename = os.path.join(self.folder_name, filename)
        self.h5_file = h5py.File(filename, 'r')
        return self.h5_file
"""

"""
PS_dataset = ProstateDataset_normalised('../Prostate_dataset', '/Users/iani/Documents/MRes/Project/Code_GPU/rectum_labels.csv',mode= 'train', augment = True)
PS_dataloader = DataLoader(PS_dataset,  batch_size = 1, shuffle =  True) #, sampler = RandomSampler) 

from matplotlib import pyplot as plt
for idx, img in enumerate(PS_dataloader):
    mri_ =  img[0]
    prostate = img[1]
    tumour = img[2]
    rectum_pos = img[3]
    fig = plt.figure()
    plt.imshow(np.squeeze(mri_)[:,:, 0], cmap = 'gray', aspect = 'equal')
    
    fig = plt.figure()
    plt.imshow(np.squeeze(mri_)[:,:, 30], cmap = 'gray', aspect = 'equal')
    
    fig = plt.figure()
    plt.imshow(np.squeeze(mri_)[:,:, 69], cmap = 'gray', aspect = 'equal')
    

    plt.show()

    fig,axes = plt.subplots(1,3)
    axes[0].imshow(np.squeeze(mri_)[:,:, 0], cmap = 'gray', aspect = 'equal')
    axes[0].set_title("MRI slice")
    axes[1].imshow(np.squeeze(prostate)[:,:, 0], cmap = 'gray', aspect = 'equal')
    axes[1].set_title("Prostate mask")
    axes[2].imshow(np.squeeze(tumour)[:,:, 0], cmap = 'gray', aspect = 'equal')
    axes[2].set_title("Tumour mask")

plt.show()
print('chicken')


from matplotlib import pyplot as plt
fig,axes = plt.subplots(1,3)
axes[0].imshow(np.squeeze(mri_)[60:110,88,:], cmap = 'gray', aspect = 'equal')
axes[0].set_title("MRI slice")
axes[1].imshow(np.squeeze(prostate)[60:110,88,:], cmap = 'gray', aspect = 'equal')
axes[1].set_title("Prostate mask")
axes[2].imshow(np.squeeze(tumour)[60:110,88,:], cmap = 'gray', aspect = 'equal')
axes[2].set_title("Tumour mask")
plt.show()


"""