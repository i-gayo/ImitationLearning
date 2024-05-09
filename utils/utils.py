import numpy as np 
from matplotlib import pyplot as plt 
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader, RandomSampler 
import pandas as pd 
import os 
import h5py
import torch 
from networks.networks import * 
from torch.utils.tensorboard import SummaryWriter


def extract_volume_params(binary_mask):
    
    """ 
    A function that extracts the parameters of the tumour masks: 
        - Bounding box
        - Centroid
    
    Parameters:
    ------------
    binary_mask : Volume of binary masks

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


    total_area = len(idx_nonzero[0]) #Number of non-zero pixels 
    z_centre = np.round(np.sum(idx_nonzero[2])/total_area)
    y_centre = np.round(np.sum(idx_nonzero[0])/total_area)
    x_centre = np.round(np.sum(idx_nonzero[1])/total_area)
        
    #In pixel coordinates 
    y_dif, x_dif, z_dif = max_vals[:] - min_vals[:]
    UL_corner = copy.deepcopy(min_vals) #Upper left coordinates of the bounding box 
    LR_corner = copy.deepcopy(max_vals) #Lower right corodinates of bounding box 

    #Centre-ing coordinates on rectum
    UL_corner = UL_corner[[1,0,2]] #- self.rectum_position
    LR_corner = LR_corner[[1,0,2]] #- self.rectum_position

    #Bounding box values : coordinates of upper left corner (closest to slice 0) + width, height, depth 
    bb_values = [UL_corner, x_dif, y_dif, z_dif, LR_corner] 

    centroid = np.asarray([x_centre, y_centre, z_centre]).astype(int)

    return bb_values, centroid 

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
        if isinstance(lesion_mask_path, tuple):
            lesion_mask = sitk.ReadImage(lesion_mask_path[0], sitk.sitkUInt8) #uncomment for multipatient_env_v2
        else:
            lesion_mask = sitk.ReadImage(lesion_mask_path, sitk.sitkUInt8)

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

class GridLabeller:
    
    """
    A class that deals with labelling of best positions on the grid (best on closest to lesion centroid)

    Functions included within labeller:
    -----------------
    overlay_grid: overlays projections of prostate, lesion and template grid


    """

    def overlay_grid(self, prostate_mask, lesion_mask, prostate_centre):
        """
        Display grid pos on top of the prostate images, centred on the prostate gland 

        Parameters:
        ---------------
        prostate_mask : (200x200x96) binary array
        lesion_mask: (200x200x96) array of multiple lesions
        prostate_centre : (3,1) array of centre of prostate 
        """
        prostate_proj = self.get_projection(prostate_mask)
        lesion_proj = self.get_projection(lesion_mask)

        grid, grid_single_val = self.obtain_grid(prostate_centre)

        # Arbritrary values 5 and 10 chosen to have different scale for lesion, grid 
        img_overlay = prostate_proj + lesion_proj*5 + grid*10 

        return img_overlay, grid, grid_single_val 

    def obtain_grid(self, prostate_centroid):
        """
        Obtains a grid centred on the prostate centre x,y 

        Parameters:
        ----------
        prostate_centroid : (3x1) array of y,x,z coords of prostate 

        Returns:
        ---------
        grid_array : + shapes are added to grid positions
        grid_single_val : 1 where grid points are, 0 otherwise (ie dots instead of + shapes)
        """

        grid_array = np.zeros((200,200))   
        grid_single_val = np.zeros((200,200))
        for i in range(-60, 65, 10):
            for j in range(-60, 65, 10):
                grid_array[prostate_centroid[0] +j  - 1:prostate_centroid[0]+j+ 1,prostate_centroid[1]+i - 1:prostate_centroid[1] +i + 1 ] = 1
                grid_single_val[prostate_centroid[0]+j , prostate_centroid[1] +i] = 1

        return grid_array, grid_single_val

    def get_labels(self, tumour_centroids, grid, num_needles = 3):
        """
        A function that finds the best position on the grid based on closest position  

        Parameters:
        -----------
        tumour_centroids : centre of each lesion, in the form of a list 
        grid : (200x200) array where 1 is where grid points are located, 0 otherwise
        num_needles : how many needles to fire per lesion (usually between 3-5)

        Returns:
        ---------
        fired_grid : (200x200) where we fire lesion only 
        fired_points: (num_lesions x num_needles) index array of each needle position (between 0-169 possible positions)
        """


        grid_coords = np.array(np.where(grid != 0)) #grid_coords where non-zero occurs 
        num_lesions = len(tumour_centroids)
        fired_points = np.zeros((num_lesions, num_needles))

        for idx, lesion_centre in enumerate(tumour_centroids):
        
            centre_point = np.reshape(lesion_centre[0:2], (2,-1))
            
            # Compute distance from centre of lesion to grid coords 
            dif_to_centroid = grid_coords - centre_point
            dist_to_centroid = np.linalg.norm(dif_to_centroid, axis = 0)
            
            # Rule : Choose 2 closest to the centroid of lesion 
            closest_points = np.argsort(dist_to_centroid)
            fired_points[idx, :] = closest_points[0:num_needles]
        
        fired_grid = self.get_fired_grid(fired_points, grid_coords)
        small_grid = self.get_small_grid(np.concatenate(fired_points))

        return fired_points, fired_grid, small_grid

    def get_projection(self, binary_mask):
        """
        Obtains 2D projections of binary and tumour mask 

        Parameters:
        -----------
        binary_mask (ndarray or torch.tensor): 200x200x96
        type (str): lesion or prostate 
        """


        projection = np.max(binary_mask, axis = 2)

        return projection 

    def get_fired_grid(self, fired_points, grid_coords):
        """
        A visual grid with all the fired points marked as 1, and elsewhere 0
        """
        
        # Plotting only the fired grid points 
        all_fired_points = np.concatenate(fired_points)
        fired_grid = np.zeros((200,200))  # 1 where we want to fire a needle ie 2 closest grid poitns to centre of lesion 
        for point in all_fired_points:
            coords = grid_coords[:,int(point)]
            #print(coords)
            fired_grid[coords[1] - 1 :coords[1] +1 , coords[0] -1 : coords[0] +1] = 1

        return fired_grid 
    
    def get_small_grid(self, labels):
        """
        Obtains a small 30x30 grid of the labelled grid points 
        """

        # Coords for each of 13 x 13 grids -> convert to 30x30 grid 
        nx_13, ny_13 = np.meshgrid(np.arange(0, 13), np.arange(0,13))
        coords = np.array([np.reshape(nx_13, -1), np.reshape(ny_13, -1)])

        grid_label = np.zeros((30,30))

        for points in labels:
            points = int(points)
            grid_label[int(2*(coords[0,points]+1)), int(2*(coords[1,points]+1))] = 1
        
        return grid_label 

### Datasets and dataloaders 

class Image_dataloader(Dataset):

    def __init__(self, folder_name, csv_path, mode = 'train', use_all = False):
        
        self.folder_name = folder_name
        self.mode = mode
        #self.rectum_df = pd.read_csv(rectum_file)
        self.all_file_names = self._get_patient_list(os.path.join(self.folder_name, 'lesion'))

        # Obtain list of patient names with multiple lesions -> change to path name
        #df_dataset = pd.read_csv('./patient_data_multiple_lesions.csv')
        df_dataset = pd.read_csv(csv_path)
        #Filter out patients >=5 lesions 
        patients_w5 = np.where(df_dataset[' num_lesions'] >= 5)[0] # save these indices for next time!!!
    
        # Remove patients where lesions >5 as these are incorrectly labelled!!
        df_dataset = df_dataset.drop(df_dataset.index[patients_w5])
        self.all_file_names = df_dataset['patient_name'].tolist()
        self.num_lesions = df_dataset[' num_lesions'].tolist()

        # Train with all patients 
        if use_all:

            size_dataset = len(self.all_file_names)

            train_len = int(size_dataset * 0.7) 
            # test_len = int(size_dataset * 0.2) 
            test_len = 0
            val_len = size_dataset - (train_len + test_len)

            # both test and val have simila rnumber of lesions (mean = 2.4 lesions)
            self.all_names = self.all_file_names
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
 

        self.dataset_len = {'train' : train_len, 'test': test_len, 'val' : val_len, 'all' : size_dataset}

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

    def _get_rectum_pos(self, patient_name):
        
        # Finding x,y,z of values with file name 
        #patient_name = 'Patient482687956_study_0.nii.gz' 
        patient_data = self.rectum_df[self.rectum_df['file_name'].str.contains(patient_name)]
        rectum_pos = [patient_data[pos].values[0] for pos in ['x', 'y', 'z']] 
        
        return rectum_pos 

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

        elif self.mode == 'all':
            patient_name = self.all_names[idx]

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

class SL_dataset(Dataset):

    def __init__(self, folder_name, labels_path = 'grid_labels.h5', mode = 'train'):

        self.folder_name = folder_name
        self.mode = mode
        self.labels_path = labels_path

        #self.all_file_names = self._get_patient_list(os.path.join(self.folder_name, 'lesion'))

        # Obtain list of patient names with multiple lesions -> change to path name
        #df_dataset = pd.read_csv('/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv')
        df_dataset = pd.read_csv('/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv')
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
        lesion_mask = np.transpose(self._normalise(read_img(os.path.join(self.lesion_folder, patient_name))), [1, 2, 0])
        prostate_mask = np.transpose(self._normalise(read_img(os.path.join(self.prostate_folder, patient_name))), [1, 2, 0])
        
        # Obtain combined prostate_lesion and turn into torch tensor
        combined_mask = (torch.from_numpy(self.get_img_mask(prostate_mask, lesion_mask))[0::2, 0::2, 0::2])/2
        
        # Get file path name 
        sitk_img_path = os.path.join(self.lesion_folder, patient_name)

        # Read grid image label 
        grid_file = self.grid_labels[patient_name]
        grid_img = grid_file['small_fired_grid'] # 30 x 30 # labels 
        large_grid_img = torch.tensor(np.array(grid_file['fired_grid'])) # 200 x 200 (same input as image)
        base_apex_grid = torch.tensor(np.array(grid_file['simple_grid']))

        # Pad_img to 256 x 256 
        pad_mask = (0, 0, 14, 14, 14, 14)
        pad_grid = (28, 28, 28, 28)
        
        combined_mask = torch.nn.functional.pad(combined_mask, pad_mask, value = 0)
        large_grid_img = torch.nn.functional.pad(large_grid_img, pad_grid, value = 0)

        # Squeeze to dimensions (channel dim + torch dim)
        combined_mask = torch.unsqueeze(combined_mask, axis = (0))
        large_grid_img = torch.unsqueeze(large_grid_img, axis = (0))
        base_apex_grid = torch.unsqueeze(base_apex_grid, axis = (0))

        return combined_mask, patient_name, large_grid_img, base_apex_grid
    
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

class TimeStep_data(Dataset):

    def __init__(self, folder_name, labels_path = 'action_labels.h5', mode = 'train', finetune = False):

        self.folder_name = folder_name
        self.mode = mode
        self.labels_path = labels_path
        self.finetune = finetune

        # Obtain list of patient names with multiple lesions -> change to path name
        #df_dataset = pd.read_csv('/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv')
        df_dataset = pd.read_csv('/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv')
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
            sampled_grid = np.array(all_grids[random_idx - 2 :random_idx+1 , :])
            
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

class TimeStep_data_test(Dataset):

    def __init__(self, folder_name, labels_path = 'action_labels.h5', mode = 'train'):

        self.folder_name = folder_name
        self.mode = mode
        self.labels_path = labels_path

        # Obtain list of patient names with multiple lesions -> change to path name
        #df_dataset = pd.read_csv('/raid/candi/Iani/Biopsy_RL/patient_data_multiple_lesions.csv')
        df_dataset = pd.read_csv('/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv')
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
        #random_idx = np.random.choice(np.arange(0, NUM_ACTIONS-2))
        #print(f'Len of actions : {NUM_ACTIONS} random idx : {random_idx}')
        random_idx = 0 
        #sampled_actions = all_actions[random_idx:random_idx+3, :]
        
        final_action = all_actions[random_idx,:] # Only consider final action to be estimated
        sampled_grid = all_grids[random_idx:random_idx +5, :]

        # add grid to patient mask (lesion + prostate mask)
        combined_grid = np.zeros((100,100,125))
        for i in range(5): 
            idx = i*25
            combined_grid[:,:,idx] = sampled_grid[i,:,:]
            combined_grid[:,:,idx+1: (i+1)*25] = combined_mask 
            #print(f'Idx : {idx}')
            #print(f' Idx : {idx+1} : {(i+1)*25}')

        # Combined grid : Contains template grid pos chosen at time steps t-3 : T
        # Final action : Action to take from time step T 
        combined_grid = torch.unsqueeze(torch.tensor(combined_grid), axis = 0)
        final_action = torch.tensor(self._normalise_actions(final_action))#, axis = 0)
        #all_actions = final_action 
        #final_action = all_actions[random_idx,:]

        # Combined actions 
        normalised_all_actions =  np.stack([self._normalise_actions(action) for action in all_actions])
        
        return combined_grid, normalised_all_actions, lesion_mask, all_grids
    
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


### Training scripts 

def validate(val_dataloader, model, use_cuda = True, save_path = 'model_1', save_images = False, metric = 'rmse'):

    # Set to evaluation mode 
    model.eval()
    acc_vals_eval = [] 
    loss_vals_eval = [] 

    loss_fn = torch.nn.BCEWithLogitsLoss()
    for idx, (images, patient_name, large_grid_img, labels) in enumerate(val_dataloader):
        
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            output = model(images)
            loss = loss_fn(output[:,:,0:13, 0:13, :].float(), labels.float()) 
            
            if metric == 'acc':
                acc, precision, recall  = compute_accuracy(output[:,:,0:13, 0:13, :].float(), labels.float())                
            else:
                acc = compute_rmse(loss)

            loss_vals_eval.append(loss.item())
            acc_vals_eval.append(acc)

        if save_images:
            # Save image, labels and outputs into h5py files
            img_name = patient_name[0].split(".")[0] + '_rectum_PRED.nrrd'
            img_path = os.path.join(save_path, img_name)
            sitk.WriteImage(sitk.GetImageFromArray(images.cpu()), img_path)
    
    with torch.no_grad():
        mean_acc = torch.mean(torch.FloatTensor(acc_vals_eval))
        mean_loss = torch.mean(torch.FloatTensor(loss_vals_eval))

    return mean_loss, mean_acc

def validate_pertimestep(val_dataloader, model, use_cuda = True, save_path = 'model_1', save_images = False, metric = 'rmse'):

    # Set to evaluation mode 
    model.eval()
    acc_vals_eval = [] 
    loss_vals_eval = [] 

    loss_fn = torch.nn.BCEWithLogitsLoss()
    for idx, (images, labels) in enumerate(val_dataloader):
        
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            output = model(images)
            loss = loss_fn(output.float(), labels.float()) 
            
            if metric == 'acc':
                acc, precision, recall  = compute_accuracy(output[:,:,0:13, 0:13, :].float(), labels.float())                
            else:
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

def dice_loss(pred_mask, gt_mask):
    '''
    y_pred, y_true -> [N, C=1, D, H, W]
    '''

    # Assuming cropping occurs outside 
    gt_cropped = gt_mask   
    pred_cropped = pred_mask

    numerator = torch.sum(gt_cropped*pred_cropped, dim=(2,3,4)) * 2
    denominator = torch.sum(gt_cropped, dim=(2,3,4)) + torch.sum(pred_cropped, dim=(2,3,4)) + 1e-6

    dice_loss = torch.mean(1. - (numerator / denominator))
    
    return dice_loss

def train(model, train_dataloader, val_dataloader, num_epochs = 10, use_cuda = False, save_folder = 'model_1', loss_fn_str = 'BCE'):
    
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
    if loss_fn_str == 'BCE':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

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
        fp.write('''epoch_num, bce_loss, accuracy''')
        fp.write('\n')

    with open(csv_val_path, 'w') as fp:
        fp.write('''epoch_num, bce_loss, accurcay''')
        fp.write('\n')

    for epoch_no in range(num_epochs):
        
        acc_vals = []
        loss_vals = [] 

        # Initialise training loop
        for idx, (images, patient_name, large_grid_img, labels) in enumerate(train_dataloader):
            
            #print(f'\n Idx train : {idx}')

            # Move to GPU 
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()

            # Training steps 
            optimiser.zero_grad()
            pred_masks = model(images)  # Obtain predicted segmentation masks 
            loss = loss_fn(pred_masks[:,:,0:13, 0:13, :].float(), labels.float()) # Compute the masks 
            dice = dice_loss(pred_masks[:,:,0:13, 0:13, :].float(), labels.float())
            loss.backward() # Backward propagation of gradients with respect to loss 
            optimiser.step() 

            # Debugging print functions
            #print(f'Unique vals : {np.unique(images.detach().cpu())}')
            #print(f'Unique vals : {np.unique(pred_masks.detach().cpu())}')
            #print(f'Unique vals label: {np.unique(labels.detach().cpu())}')
            #print(f'Loss : {loss.item()}')

            # Compute metrics for each mini batch and append to statistics for each epoch
            with torch.no_grad():
                accuracy, precision, recall  = compute_accuracy(pred_masks[:,:,0:13, 0:13, :], labels)
                #print(f'Accuracy : {accuracy.item()}')
                acc_vals.append(accuracy)
            
            loss_vals.append(loss.item())

            # Print loss every nth minibatch and dice score 
            #if idx % freq_print == 0: 
            #    print(f'Epoch {epoch_no} minibatch {idx} : loss : {loss.item():05f}, acc score : {acc.item():05f}')
            
        # Obtain mean dice loss and acc over this epoch, save to tensorboard 
        acc_epoch = torch.mean(torch.tensor(acc_vals))
        loss_epoch = torch.mean(torch.tensor(loss_vals))

        print(f'\n Epoch : {epoch_no} Average loss : {loss_epoch:5f} average acc {acc_epoch:5f}')

        with open(csv_train_path, 'a') as fp: 
            loss_points = np.stack([epoch_no, loss_epoch, acc_epoch]).reshape(1,-1)
            np.savetxt(fp, loss_points, '%s', delimiter =",")

        # Save for all_loss_train
        all_loss_train[epoch_no] = loss_epoch
        all_acc_train[epoch_no] = acc_epoch 
        
        #Tensorboard saving 
        writer.add_scalar('Loss/train', loss_epoch, epoch_no)
        writer.add_scalar('acc/train', acc_epoch, epoch_no)
    
        # Save newest model 
        train_model_path = os.path.join(train_folder, 'train_model.pth')
        torch.save(model.state_dict(), train_model_path)

        # Validate every nth epoch and save every nth mini batch 
        if epoch_no % freq_eval == 0: 

            # Set to evaluation mode 
            if epoch_no % 100 == 0: 
                save_img = True
            else:
                save_img = False 
                
            mean_loss, mean_acc = validate(val_dataloader, model, use_cuda = use_cuda, save_path = train_folder, save_images = False)
            print(f'Validation loss for epoch {epoch_no} Average loss : {mean_loss:5f} average acc {mean_acc:5f}')
            all_loss_val.append(mean_loss)
            all_acc_val.append(mean_acc)
            
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, mean_loss, mean_acc]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

            #Tensorboard saving
            writer.add_scalar('Loss/val', mean_loss, epoch_no)
            writer.add_scalar('acc/val', mean_acc, epoch_no)

            if mean_loss < best_loss: 
                
                # Save best model as best validation model 
                val_model_path = os.path.join(train_folder, 'best_val_model.pth')
                torch.save(model.state_dict(), val_model_path)
        
                # Use as new best loss
                best_loss = mean_loss 
        
        elif epoch_no == 0: 
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, 1, 0]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")
        else:
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, all_loss_val[-1], all_acc_val[-1]]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

        #print('Chicken') 
    return all_loss_train, all_loss_val, all_acc_train, all_acc_val 

def train_pertimestep(model, train_dataloader, val_dataloader, num_epochs = 10, use_cuda = False, save_folder = 'model_1', loss_fn_str = 'MSE'):
    
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
    if loss_fn_str == 'MSE':
        loss_fn = torch.nn.MSELoss()
        print('loss fn : MSE Loss')
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
        print('loss fn : BCE Loss')

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
        
        acc_vals = []
        loss_vals = [] 

        # Initialise training loop
        for idx, (images, labels) in enumerate(train_dataloader):
            
            #print(f'\n Idx train : {idx}')

            # Move to GPU 
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()

            # Training steps 
            optimiser.zero_grad()
            pred_actions = model(images)  # Obtain predicted segmentation masks 
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

            # Set to evaluation mode 
            if epoch_no % 100 == 0: 
                save_img = True
            else:
                save_img = False 
                
            mean_loss, mean_acc = validate_pertimestep(val_dataloader, model, use_cuda = use_cuda, save_path = train_folder, save_images = False)
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

def compute_rmse(mse):
    """
    Computes rmse as a metric to be reported, but not to be used for comptuation of loss fns 
    """

    with torch.no_grad():
        mse_copy = torch.clone(mse.detach())
        rmse = torch.sqrt(mse_copy)

    return rmse 

def compute_accuracy(predicted, gt):
    """
    Computes how many pixels are the same / 1 - cross entropy loss 
    
    Compute how many pixels are both 1 in the image!!! 
    """

    # Convert to view -1 
    with torch.no_grad():
        sigmoid_layer = torch.nn.Sigmoid()
        pred_vals = sigmoid_layer(predicted)
        pred_vals = predicted.reshape(-1)
        pred_vals[pred_vals >= 0.5] = 1 # use threshold of 0.5 for segmentation 
        gt_vals = gt.reshape(-1)
        num_pixels = len(gt_vals)
        
        # Obtaining TP, FP, FN, TN 
        tp = int(torch.sum((gt_vals == 1) * (pred_vals == 1)))
        tn = int(torch.sum((gt_vals == 0) * (pred_vals == 0)))
        fp = int(torch.sum((gt_vals == 0) * (pred_vals == 1)))
        fn = int(torch.sum((gt_vals == 1) * (pred_vals == 0)))

        # Obtaining accuracy, precision and recall 
        accuracy = (tp + tn) / (num_pixels + 1e-5) # how many pixels were correctly identified as 1 or 0 
        precision = tp / (tp + fp + 1e-6) # how many predicted positive values were actually positive 
        recall = tp / (tp + fn + 1e-6) # how many positive values were actually found 

    return accuracy, precision, recall 

def compute_inv_bce(bce_loss):

    with torch.no_grad():
        inv_bce = 1 - bce_loss 

    return inv_bce # (ie 1 - BCE loss)

def compute_bce(predicted, gt):
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        bce = loss_fn(predicted.float(), gt.float())

    return bce 

def dice_loss(pred_mask, gt_mask):
    '''
    y_pred, y_true -> [N, C=1, D, H, W]
    '''

    # Assuming cropping occurs outside 
    gt_cropped = gt_mask   
    pred_cropped = pred_mask

    numerator = torch.sum(gt_cropped*pred_cropped, dim=(2,3,4)) * 2
    denominator = torch.sum(gt_cropped, dim=(2,3,4)) + torch.sum(pred_cropped, dim=(2,3,4)) + 1e-6

    dice_loss = torch.mean(1. - (numerator / denominator))
    
    return dice_loss

### Biopsy metrics

def compute_hit_rate(obs, action):
    """
    Using observations, compute the hit rate 
    """

    # hit if : grid position intersects with lesion 
    
    # 1. given observation and action -> what is new state? 
    # compute np.where(lesion) * np.where(grid_pos)

if __name__ == '__main__':

    file_path = '/Users/ianijirahmae/Documents/PhD_project/Biopsy_RL/grid_labels_h5.h5'
    hf = h5py.File(file_path, 'r')

    # Accessing elements in h5py file!!! 
    group_1 = hf['Patient018758409_study_0.nii.gz'] # 30 x 30 
    small_fired_grid = np.array(group_1['small_fired_grid']) #200 x 200 

    # Initialising test set
    PS_PATH = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality'
    #LABELS_PATH = '/Users/ianijirahmae/Documents/PhD_project/Biopsy_RL/grid_labels_all.h5'
    LABELS_PATH = '/Users/ianijirahmae/Documents/PhD_project/Biopsy_RL/action_labels.h5'
    
    #train_ds = SL_dataset(PS_PATH, LABELS_PATH, mode = 'train')
    train_ds = TimeStep_data(PS_PATH, LABELS_PATH)
    train_dl = DataLoader(train_ds, batch_size = 32, shuffle = True)
    grid, actions = train_ds[0]

    from networks.networks import * 
    model = ImitationNetwork()
    test_output = model(grid)
    test = train_ds[0]
    #combined_mask, patient_name, large_grid_img, base_apex_img = train_ds[1]
    #plt.imshow(torch.squeeze(torch.squeeze(large_grid_img[:,:])))


    print('Chicken')





