import numpy as np
import pandas as pd 
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, RandomSampler 
import os 
import SimpleITK as sitk
import copy 
import csv 
import h5py
#from utils import *
import torch 

def separate_actions(comb_actions, all_ids, idx):
    """
    A function that obtains actions only corresponding to lesion with idx 

    :comb_actions: 3xN action array 
    :all_ids: 3x1 action identifiers; ids for which lesion each action corresponds to 

    Returns:
    :sep_actions: 3 x num_steps where num_steps is the number of steps corresponding to chosen idx lesion 
    """

    # obtain idx of actions corresponding to lesion 
    idx_map = (all_ids == idx)[0]

    # return actions corresponding to lesion idx only 
    sep_actions = comb_actions[:,idx_map]

    return sep_actions 

def separate_masks(multiple_lesions):

    # for each lesion: 
    unique_idx = np.unique(multiple_lesions)

    num_lesion_masks = len(unique_idx) - 1

    all_masks = [] 
    for idx in range(num_lesion_masks):
        mask = multiple_lesions == idx+1 # add 1 as we don't want background 
        all_masks.append(mask)

    return all_masks 

class ActionFinder():

    def __init__(self, lesion_means, needle_coords, needle_depths, start_pos = np.array([0,0])):
        self.lesion_means = lesion_means
        self.needle_coords = needle_coords
        self.needle_depths = needle_depths 
        self.start_pos = start_pos 
        self.action_maps = {3 : np.array([2,1]), 4 : np.array([2,2]), 5 : np.array([2,2,1]), 6 : np.array([2,2,2]), \
            7 : np.array([2,2,2,1]), 8 : np.array([2,2,2,2]), 9: np.array([2,2,2,2,1]), 10: np.array([2,2,2,2,2]), \
                11: np.array([2,2,2,2,2, 1]), 12: np.array([2,2,2,2,2,2]), 13: np.array([2,2,2,2,2,2,1])}
        NUM_LESIONS = len(lesion_means)

        self.map = {}
        for idx in range(NUM_LESIONS):
            self.map[idx] = [self.lesion_means[idx], self.needle_coords[idx]]

        self.remaining_idx = [i for i in range(NUM_LESIONS)]
        self.visited_idx = []

    def find_closest_lesion(self, lesion_means, idx_map, starting_point):
        """
        lesion_means : (n,2) where n is number of lesions
        starting_point : (2,) current coordinate position on grid 

        Returns: 
        -----------------
        closest_idx: ordered idx values of closest lesion point 
        """
        
        # Distance between starting point and all remaining lesions 
        dif_vec = lesion_means - starting_point 
        dist_to_point = np.linalg.norm(dif_vec, axis = 1)

        # Sort by distance, return index of closest lesion 
        closest_idx = np.argsort(dist_to_point)[0]
        needle_mean = lesion_means[closest_idx]
        
        # Closest lesion index (using index map)
        closest_idx_mapped = idx_map[closest_idx]

        return closest_idx_mapped, needle_mean

    def find_closest_points(self, needle_coords, closest_lesion, starting_point):
        """
        Returns closest points to starting point from chosen lesion 
        """

        dif_vec = starting_point.reshape(2,1) - needle_coords[closest_lesion][:2, :]
        closest_idx = np.argsort(np.linalg.norm(dif_vec, axis = 0))

        return closest_idx 

    def get_rel_actions(self, needle_coords, closest_idx, starting_point):
        """
        Obtains relative actions to take from starting point to points on each grid
        """
        
        # Reorderd needle coords according to distance to starting point 
        reordered_points = needle_coords[:,closest_idx] # Reordered coords according to distance to previous point 
        
        # Obtain dif points (first element is starting point, next element are positions following needle coords)
        dif_points = np.zeros_like(reordered_points)
        dif_points[:,0] = starting_point
        dif_points[:, 1:] = reordered_points[:, 0:-1]
        #print(f'Reordered points : \n {reordered_points}')

        relative_actions = (reordered_points[:,:] - dif_points[:,:])
        new_start_pos = reordered_points[:,-1]

        return relative_actions, new_start_pos

    def find_actions(self, starting_point):
        """
        
        Notes:
        ---------
        self.remaining_idx : list of indexes corresponding to lesions that have not yet been visited
        self.lesion_means : mean coordinates of each lesion 

        """
        
        print(f'starting point : {starting_point}')
        needle_means = [] # needle means of remaining lesions to be visited
        idx_map = {} # maps remaining lesion index to actual needle index
        

        for map_idx, needle_idx in enumerate(self.remaining_idx): 
            needle_means.append(self.lesion_means[needle_idx])
            idx_map[map_idx] = needle_idx # Maps [0 1 2] -> [2 0 1] where needle_idx is actual idx of positions of lesions 

        # Get closest lesion 
        closest_idx, lesion_mean = self.find_closest_lesion(needle_means, idx_map, starting_point)
        #print(f'Closest_idx : {closest_idx}')
        #print(f'Starting point {starting_point}')
        #print(f'Needle coords : {needle_coords[closest_idx]}')

        # Sample 4 needle points belonging to closest lesion. Point idx is order of close-ness
        points_idx = self.find_closest_points(self.needle_coords, closest_idx, starting_point)

        # Obtain relative actions 
        relative_actions, new_start_pos = self.get_rel_actions(self.needle_coords[closest_idx], points_idx, starting_point)

        # Refine actions -> ie add firing (0 or 1) and move big actions to smaller ones (from 6 -> 2, 2, 2)
        refined_actions = self.refine_actions(relative_actions, self.needle_depths[closest_idx])
        _, num_steps = np.shape(refined_actions)
        idx_identifier = np.ones((1,num_steps))*closest_idx
        #stacked_actions = np.vstack(refined_actions, lesion_idx)

        # Save visited idx, remove from map 
        self.remaining_idx.remove(closest_idx)
        self.visited_idx.append(closest_idx)
        #print(f'Remaining idx {self.remaining_idx}')

        return refined_actions, new_start_pos, idx_identifier

    def refine_actions(self, actions, needle_depths):
        """
        
        Adds firing action delta_z
        Changes larger actions (greater than 10mm or 2 grid positions) to smaller discrete ones

        """

        num_needles = np.shape(actions)[1]

        refined_actions = [] 

        for idx in range(num_needles):

            act = actions[:,idx]
            depth = int(needle_depths[idx])

            paired_actions = [] 

            # Loop through each action x,y 
            for indv_act in act: 

                abs_indv_act = np.abs(indv_act)
                sign = np.sign(indv_act)
                
                # Loop through each x,y action 
                # If size of action is greater than 2 (ie greater than 10mm) -> split up actions to individual actions 
                if abs_indv_act > 2:
                    split_actions = sign * self.action_maps[abs_indv_act]
                else: 
                    split_actions = np.array([indv_act])
                
                paired_actions.append(split_actions)

            # Create array of split_actions : ie from [4, 3] -> [[2,2], [2,1]] to split up max movement 
            len_actions = [len(action) for action in paired_actions]
            new_actions = np.zeros([3, np.max(len_actions)])
            
            new_actions[0, 0:len_actions[0]] = paired_actions[0]
            new_actions[1, 0:len_actions[1]] = paired_actions[1]
            
            # Hit at the end of the split actions (ie when reached actual needle destination)
            new_actions[2, -1] = depth

            refined_actions.append(new_actions)
        
        # Concatenate actions
        refined_actions = np.concatenate(refined_actions, axis = 1)
        
        return refined_actions 
    
    def return_visit_order(self):
        print(f'Visited order : {self.visited_idx}')
        return self.visited_idx

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
        #lesion_mask = sitk.ReadImage(lesion_mask_path[0], sitk.sitkUInt8) #uncomment for multipatient_env_v2
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

        lesion_statistics = {'lesion_size' : lesion_size, 'lesion_bb' : lesion_bb, 'lesion_diameter' : lesion_diameter, 'lesion_centroids' : lesion_centroids} 

        # Convert centroids from mm to pixel coords if not needed in mm 
        if not self.give_centroid_in_mm: 
            lesion_centroids = [lesion_centroid * np.array([2,2,1]) for lesion_centroid in lesion_centroids]
        
        num_lesions = label_shape_filter.GetNumberOfLabels()

        return lesion_centroids, num_lesions, lesion_statistics, np.transpose(multiple_label_img, [1, 2, 0])

class Image_dataloader(Dataset):

    def __init__(self, folder_name, csv_path, mode = 'train', use_all = False):
        
        self.folder_name = folder_name
        self.mode = mode
        #self.rectum_df = pd.read_csv(rectum_file)
        self.all_file_names = self._get_patient_list(os.path.join(self.folder_name, 'lesion'))

        # Obtain list of patient names with multiple lesions -> change to path name
        #df_dataset = pd.read_csv('./patient_data_multiple_lesions.csv')
        df_dataset = pd.read_csv(csv_path)
        #self.all_file_names = df_dataset['patient_name'].tolist()
        #self.num_lesions = df_dataset[' num_lesions'].tolist()

        # Train with all patients 
        if use_all:

            size_dataset = len(self.all_file_names)

            train_len = int(size_dataset * 0.7) 
            test_len = int(size_dataset * 0.2) 
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
        img_overlay = prostate_proj*2 + lesion_proj*5 #+ grid*10 

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

    def get_labels(self, prostate_mask, tumour_centroids, grid, num_needles = 3):
        """
        A function that finds the best position on the grid based on closest position  

        Parameters:
        -----------
        tumour_centroids : centre of each lesion, in the form of a list 
        grid : (200x200) array where 1 is where grid points are located, 0 otherwise
        num_needles : how many needles to fire per lesion (usually between 3-5)

        Returns:
        ---------
        fired_points: (num_lesions x num_needles) index array of each needle position (between 0-169 possible positions)
        fired_grid : (200x200) where we fire lesion only 
        """

        # find prostate_mask depth and mid point
        prostate_mask_depths = np.where(prostate_mask !=0)[-1]
        prostate_mid_point = (np.max(prostate_mask_depths) + np.min(prostate_mask_depths)) / 2

        grid_coords = np.array(np.where(grid != 0)) #grid_coords where non-zero occurs 
        num_lesions = len(tumour_centroids)
        
        if num_lesions == 0: 
            print('Chicken')
        
        fired_points = np.zeros((num_lesions, num_needles))
        fired_depths = np.zeros((num_lesions, num_needles))

        # firing_grid with 2 points 
        simple_grid = np.zeros((13,13,2)) # first is apex, second is base 
        
        depth = [] 

        for idx, lesion_centre in enumerate(tumour_centroids):
        
            centre_point = np.reshape(lesion_centre[0:2], (2,-1))
            
            # Compute distance from centre of lesion to grid coords 
            dif_to_centroid = grid_coords - centre_point
            dist_to_centroid = np.linalg.norm(dif_to_centroid, axis = 0)
            
            # Rule : Choose n closest to the centroid of lesion 
            closest_points = np.argsort(dist_to_centroid)
            fired_points[idx, :] = closest_points[0:num_needles]
            
            # Check at what depth 
            if lesion_centre[-1] < prostate_mid_point:
                #print(f'Apex lesion: midpoint : {prostate_mid_point}, lesion_centre : {lesion_centre[-1]}')
                for i in range(num_needles):
                    depth.append(0) # base
                    fired_depths[idx, :] = 1 # base
            else:
                #print(f'Base lesion : midpoint : {prostate_mid_point}, lesion_centre : {lesion_centre[-1]}')
                for i in range(num_needles):
                    depth.append(1) # apex 
                    fired_depths[idx, :] = 2
    
        fired_grid, fired_grid_depth = self.get_fired_grid(fired_points, grid_coords, depth)
        small_grid, small_grid_depth = self.get_small_grid(np.concatenate(fired_points), depth)
        simple_grid = self.get_simple_grid(np.concatenate(fired_points), depth)
        
        return fired_points, fired_grid, fired_grid_depth, small_grid, small_grid_depth, simple_grid, fired_depths

    def get_labels_per_step(self, prostate_mask, tumour_centroids, grid, num_needles = 3):
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

        grid per time step (with positions)
        
        """

        # find prostate_mask depth and mid point
        prostate_mask_depths = np.where(prostate_mask !=0)[-1]
        prostate_mid_point = (np.max(prostate_mask_depths) + np.min(prostate_mask_depths)) / 2

        grid_coords = np.array(np.where(grid != 0)) #grid_coords where non-zero occurs 
        num_lesions = len(tumour_centroids)
        
        if num_lesions == 0: 
            print('Chicken')
        
        fired_points = np.zeros((num_lesions, num_needles))

        # firing_grid with 2 points 
        simple_grid = np.zeros((13,13,2)) # first is apex, second is base 
        
        depth = [] 

        for idx, lesion_centre in enumerate(tumour_centroids):
        
            centre_point = np.reshape(lesion_centre[0:2], (2,-1))
            
            # Compute distance from centre of lesion to grid coords 
            dif_to_centroid = grid_coords - centre_point
            dist_to_centroid = np.linalg.norm(dif_to_centroid, axis = 0)
            
            # Rule : Choose n closest to the centroid of lesion 
            closest_points = np.argsort(dist_to_centroid)
            fired_points[idx, :] = closest_points[0:num_needles]
            
            # Check at what depth 
            if lesion_centre[-1] < prostate_mid_point:
                #print(f'Apex lesion: midpoint : {prostate_mid_point}, lesion_centre : {lesion_centre[-1]}')
                for i in range(num_needles):
                    depth.append(0) # base
            else:
                #print(f'Base lesion : midpoint : {prostate_mid_point}, lesion_centre : {lesion_centre[-1]}')
                for i in range(num_needles):
                    depth.append(1) # apex 
    
        fired_grid, fired_grid_depth = self.get_fired_grid(fired_points, grid_coords, depth)
        small_grid, small_grid_depth = self.get_small_grid(np.concatenate(fired_points), depth)
        simple_grid = self.get_simple_grid(np.concatenate(fired_points), depth)
        
        return fired_points, fired_grid, fired_grid_depth, small_grid, small_grid_depth, simple_grid


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

    def get_fired_grid(self, fired_points, grid_coords, depths):
        """
        A visual grid with all the fired points marked as 1, and elsewhere 0
        """
        
        # Plotting only the fired grid points 
        all_fired_points = np.concatenate(fired_points)
        fired_grid = np.zeros((200,200))  # 1 where we want to fire a needle ie 2 closest grid poitns to centre of lesion 
        fired_grid_depth = np.zeros((200,200,2)) # apex and base separated 

        for point, depth in zip(all_fired_points, depths):
            coords = grid_coords[:,int(point)]
            #print(coords)
            fired_grid[coords[1] - 1 :coords[1] +1 , coords[0] -1 : coords[0] +1] = 1
            fired_grid_depth[coords[1] - 1 :coords[1] +1 , coords[0] -1 : coords[0] +1, depth] = 1

        return fired_grid, fired_grid_depth
    
    def get_small_grid(self, labels, depths):
        """
        Obtains a small 30x30 grid of the labelled grid points 
        """

        # Coords for each of 13 x 13 grids -> convert to 30x30 grid 
        nx_13, ny_13 = np.meshgrid(np.arange(0, 13), np.arange(0,13))
        coords = np.array([np.reshape(nx_13, -1), np.reshape(ny_13, -1)])

        grid_label = np.zeros((30,30))
        grid_label_depth = np.zeros((30,30,2))

        for points, depth in zip(labels, depths):
            points = int(points)
            grid_label[int(2*(coords[0,points]+1)), int(2*(coords[1,points]+1))] = 1
            grid_label_depth[int(2*(coords[0,points]+1)), int(2*(coords[1,points]+1)), depth] = 1

        return grid_label, grid_label_depth 

    def get_simple_grid(self, labels, depth):
        """
        Using label of closest points
        """
        
        simple_grid = np.zeros((13,13,2))

        nx_13, ny_13 = np.meshgrid(np.arange(0, 13), np.arange(0,13))
        coords = np.array([np.reshape(nx_13, -1), np.reshape(ny_13, -1)])

        for points, depth_val in zip(labels, depth):
            simple_grid[int(coords[0, int(points)]), int(coords[1, int(points)]), depth_val] = 1 

        return simple_grid

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

        depth_map = {0 : 0, 1 : int(0.5*max_depth), 2 : max_depth}
        depth = depth_map[int(action[2])]

        if int(action[2]) == 2:
            print('chicekn')

        if depth != 0:
            needle_vol[y_grid_pos-1:y_grid_pos+ 2, x_grid_pos-1:x_grid_pos+2, 0:depth ] = 1

        return needle_vol 

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

def get_needle_obs(needle_coords):
    """
    A function that obtains needle trajectory, given chosen grid position 

    Parameters:
    -----------
    needle_coords : x, y, z where z is [0, 1, 2] where 0 is non-fired ie empty needle, 1 is base, 2 is apex 

    Returns:
    ----------
    needle_obs : 100 x 100 x 24 array of needle position 

    """

    depth = 24 
    # apex : up to halfway (apex hit)
    # base : up to fullway (base hit)
    depth_map = {'0' : 0, '1' : int(0.5*depth), '2' : depth}

    # initialise empty obs 
    obs = np.zeros((100, 100, 24))

    # using needle x,y define obs
    x_coords = 5#TODO
    y_coords = 5#TODO 
    z_depth = depth_map[needle_coords[2]]

    # 
    if z_depth != 0:
        obs[y_coords-1 : y_coords +2, x_coords - 1 : x_coords +2, 0:z_depth] = 1

    return obs 

if __name__ == '__main__':

    # Code extracts the best positions to place needles into (ie best grid positions)
    ps_path = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality'
    csv_path = '/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv'
    
    labels_path = 'ACTION_OBS_LABELS_SINGLE.h5'
    # H5PY Dataset for saving labels : change file name to what is convenient for you
    hf = h5py.File(labels_path, 'w')     

    # Define dataloader and lesion labellers : use to load patietn volumes in 
    PS_dataset_train = Image_dataloader(ps_path, csv_path, use_all = True, mode  = 'all')
    Data_sampler_train = DataSampler(PS_dataset_train)
    NUM_DATASET = len(PS_dataset_train)

    # Classes used to obtain labels : 
    lesion_labeller = LabelLesions() # obtains lesion centroids 
    grid_labeller = GridLabeller() # obtains needle grid positions for each lesion 
    grid_creater = GridArray()

    all_max_depth = []   

    for patient_idx, (mri_vol, prostate_mask, tumour_mask, tumour_mask_sitk, rectum_pos, patient_name) in enumerate(PS_dataset_train):

        empty_tumour_mask = (len(np.unique(tumour_mask)) == 1) # in tumour mask only 0s which means no tumour is present 

        if empty_tumour_mask:
            
            # print for references, but do not use to obtain labels from 
            print(patient_name)
        
        else:
            
            ### 1. OBTAIN LESION CENTROIDS USING LABELLESIONS CLASS ###
            tumour_centroids, num_lesions, tumour_statistics, multiple_label_img = lesion_labeller(tumour_mask_sitk)
            bb_centroid, prostate_centroid = extract_volume_params(prostate_mask)

            # downsampled prostate mask 
            ds_prostate_mask = prostate_mask[::2,::2,::4]
            max_depth = np.max(np.where(ds_prostate_mask == 1)[-1])
            all_max_depth.append(int(max_depth))

            print(f"NUM LESIONS {num_lesions}")
            ### 2. FIND 4 NEEDLE GRID POSITIONS PER LESION USING GRIDLABELLER CLASS : grid_labels gives index of each needle position   ###
            # Obtain image projections of grid overlaid with lesion and prostate 
            img_overlay, grid, grid_points = grid_labeller.overlay_grid(prostate_mask, multiple_label_img, prostate_centroid)
            lesion_overlay = grid_labeller.get_projection(multiple_label_img)

            # Obtain classification grid and binary array of 1s, 0s of best grid positions) 
            grid_labels, fired_grid, fired_grid_depth, small_fired_grid, small_fired_grid_depth, simple_grid, fired_depths = grid_labeller.get_labels(prostate_mask, tumour_centroids, grid_points, num_needles = 4) 
            combined_grid = np.sum(simple_grid, axis = 2)

            #test_all_mask = separate_masks(multiple_label_img)
                    
            
            #print('chicekn')
            
            # To find coords for each lesion 
            # coord system : -6,6 left to right and -6,6 bottom to top of grid 
            x_coords, y_coords = np.meshgrid(np.arange(-6,7,1), np.arange(-6, 7, 1))
            x_ = x_coords.reshape(-1)
            y_ = y_coords.reshape(-1)
            
            # Finding coordinates of needle points per lesion 
            needle_coords = [] 
            needle_depths = [] 

            # Iterate through each lesion, obtain grid positions of each needle fired 
            for idx, points in enumerate((grid_labels)):
                idx_vals = [int(idx) for idx in points] # convert from float to int 
                needle_coords.append(np.array([x_[idx_vals], y_[idx_vals]])) # obtain x,y coordinate on grid 
                needle_depths.append(fired_depths[idx,:])

            needle_means = [np.mean(points[0:2], axis = 1) for points in needle_coords]  # mean coords of each needle points per lesion 

            print(f'Needle means \n {needle_means}')

            ### 3. OBTAIN ACTIONS PER TIME STEP USING ACTIONFINDER CLASS ###
  
            all_actions = []
            all_identifiers = [] 
            all_needle_vol = [] 

            starting_point = np.array([0,0]) #initialise starting point as centre of grid 
            action_mapper = ActionFinder(needle_means, needle_coords, needle_depths, starting_point)

            
            # Note for single lesion cases : always start from centre; for complete trajectoreis, change _ to starting_point to save last position as new startign position. 
            #starting_point = np.array([0,0]) #initialise starting point as centre of grid 
            
            # Find actions for each lesion 
            for idx_l in range(num_lesions):
                rel_actions, _, idx_identifier = action_mapper.find_actions(starting_point)

                # change starting_point to (0,0) every time 

                print(rel_actions)
                all_actions.append(rel_actions)
                all_identifiers.append(idx_identifier)

            print('chicken')
            all_actions = np.concatenate(all_actions, axis =1)
            all_identifiers = np.concatenate(all_identifiers, axis = 1)
            visited_idx = action_mapper.return_visit_order() 

            # obtain action
            #test_actions = separate_actions(all_actions1, all_identifiers1, 2)


            ### 4. OBTAIN STATES/OBSERVATIONS PER TIME STEP USING GRIDARRAY CLASS ###
            # Obtain updated grid position at each time step : 100 x 100 grid. 
            # Note : intensity = 0.25 for current position, 0.5 for non-fired but previously visited position and 1 for fired previously visited position. 0 otherwise 

            NUM_ACTIONS = np.shape(all_actions)[1]
            all_grids = [] 
            all_current_pos = [] 

            # Starting pos 
            current_pos = np.array([0.,0.,0.])

            for idx in range(NUM_ACTIONS):
                
                idx_identifier = all_identifiers[0][idx]

                # iterate through each time step and obtain actions at each time step 
                action_set = all_actions[:,idx]
                all_current_pos.append(copy.deepcopy(current_pos))
                #print(f'Action set : {action_set}')
                
                # First position, current pos is (0,0) 
                if idx == 0: 
                    first_point = True 
                    grid_array = None 

                    template_grid = np.zeros((100,100))
                    template_grid[50,50] = 0.25 # current position = 0.25 
                    all_grids.append(template_grid)
                    
                    # create needle vol 
                    needle_vol = grid_creater.create_needle_vol(current_pos, max_depth)
                else: 
                    # Update grid array 
                    grid_array, grid_his = grid_creater.create_grid_array(current_pos, grid_array)
                    all_grids.append(grid_array)

                    needle_vol = grid_creater.create_needle_vol(current_pos, max_depth)

                # Update new position 
                change_lesions = (idx_identifier != all_identifiers[0][idx-1]) and (idx != 0)
                if change_lesions: # start from the middle 
                    print('changed lesions')
                    current_pos[2] = 0 + action_set[2]
                    current_pos[0:2] = np.array([0,0]) + action_set[0:2]

                    print(f'Current pos: {current_pos}')
                else: #if the same as previous point, keep adding starting point, else: 
                    current_pos[2] = action_set[2] # z position : 0 if not-fired, 1 if fired 
                    current_pos[0:2] += action_set[0:2] # new_pos = current_pos + (delta_x, delta_y)
                    all_needle_vol.append(needle_vol)
                    print(f'Current pos: {current_pos}')
            
            # Stack all actions and observations 
            all_grids_array = np.stack(all_grids)
            all_actions = np.transpose(all_actions)
            all_pos = np.stack(all_current_pos)
            all_vol = np.stack(all_needle_vol)

            print('Chicken')

            ### 5. SAVE OBSERVATION AND ACTIONS TO H5PY FILE ###
            file_path = tumour_mask_sitk.split('/')[-1]
            grid_vals = np.concatenate(grid_labels)

            # Save to h5py file 
            group_folder = hf.create_group(file_path)

            # save per-timestep actions 
            group_folder.create_dataset('all_actions', data = all_actions) # actions at each time step from 0:T-1 where T is number of timesteps taken ; N x 3 
            group_folder.create_dataset('all_grids', data = all_grids_array) # grid arrays at each time step from 0:T-1 where T is number of timesteps taken ; N x 100 x 100 
            group_folder.create_dataset('fired_grid', data = fired_grid) # position of all grid coords to be fired ; 200 x 200 

            # save grids used for alternative supervised learning : predict all grid positions together 
            group_folder.create_dataset('fired_grid_depth', data = fired_grid_depth)  # position of each grid position ; 200 x 200 x 2 to split for base and apex firing 
            group_folder.create_dataset('simple_grid', data = simple_grid) # position of each grid position ; 13 x 13 x 2 and split into base and apex firing 
            group_folder.create_dataset('idx_labels', data = grid_vals)# index of all needle grid positions fired (from 0-168); Shape Num_lesions x 4 
            group_folder.create_dataset('lesion_size', data = tumour_statistics['lesion_size'])
            group_folder.create_dataset('lesion_centroids', data = tumour_statistics['lesion_centroids'])
            group_folder.create_dataset('lesion_img', data = lesion_overlay) # img with multiple lesion labels
            group_folder.create_dataset('action_identifiers', data = all_identifiers)           #print('\n')
            group_folder.create_dataset('multiple_lesion_img', data = multiple_label_img)         #print('\n')
            group_folder.create_dataset('current_pos', data = all_pos)  

        # READING LABELS 
        #with open('grid_labels.csv', 'a') as f:
        #    if idx !=0:
        #        f.write('\n')
        #    f.write(','.join(str(item) for item in csv_items))

    
    #

    # Close hf  
    hf.close() 


print('chicken')