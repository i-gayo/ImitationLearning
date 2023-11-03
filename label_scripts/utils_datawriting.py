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

#### FUNCTIONS ####
def find_value_changes(arr):
    diff = np.diff(arr)  # Calculate the difference between consecutive elements
    change_indices = np.where(diff != 0)[1]  # Find the indices where the difference is nonzero
    
    return change_indices


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

def separate_masks(multiple_lesions):

    # for each lesion: 
    unique_idx = np.unique(multiple_lesions)

    num_lesion_masks = len(unique_idx) - 1

    all_masks = [] 
    for idx in range(num_lesion_masks):
        mask = multiple_lesions == idx+1 # add 1 as we don't want background 
        all_masks.append(mask)

    return all_masks 

def find_closest_point(target_point, points):
    
    """
    Find the closest coordinate point from an array of coordinate points.

    Parameters:
    - target_point (tuple): The target point for which to find the closest point.
    - points (list of tuples): 2 x 169 coords array of all possible grid points 

    Returns:
    closest_index : int index between 0,169 
    closest_point : coordinates on the image grid corresponding to this point! 
    """
    # Find non-zero points: 
    
    target_point = np.array(target_point).reshape(2,1)
    #points_coords = np.array(np.where(points == 1))

    distances = np.linalg.norm(points - target_point, axis=0)
    closest_index = np.argmin(distances)
    closest_point = points[:,closest_index]

    return closest_index, closest_point

def get_wacky_points(lesion_vol, strategy = 'edges'):
    
    """
    A function that obtains 4 grid grid points corresponding to a wacky strategy 
    
    Parameters:
    ----------
    lesion_vol(ndarray) : 200 x 200 x 96 binary mask of lesion volume 
    strategy (str) : indicates which type of points to obtain : edges or box 
    
    Returns:
    ----------
    grid_points : 4 points (4 x 2) corresponding to BR, UR, UL, BL
    
    """
    # lesion 2d projection in xy plane 
    lesion_proj = np.max(lesion_vol, axis = 2)
    all_y, all_x = np.where(lesion_proj)
    mean_x = np.mean(all_x)
    needle_grid = np.zeros_like(lesion_proj)
    
    if strategy == 'edges': 
        
        #tl
        min_x = np.min(all_x)
        corr_y = np.max(all_y[all_x == min_x])
        
        #bl
        max_y = np.max(all_y[all_x < mean_x])
        corr_x = np.min(all_x[all_y == max_y]) # left most 
        
        #br
        max_x = np.max(all_x)
        corr_ymax = np.max(all_y[all_x == max_x])
        
        #tr 
        min_y = np.min(all_y[all_x >= mean_x])
        corr_xmax = np.max(all_x[all_y == min_y])
    
        needle_grid[max_y ,corr_x] = 1 #bl
        needle_grid[corr_y, min_x] = 1 # tl 
        needle_grid[corr_ymax, max_x] = 1 # br 
        needle_grid[min_y, corr_xmax] = 1 # tr
        
        # obtain array of br, tr, tl, bl 
        coords = np.array([[corr_ymax, max_x], [min_y, corr_xmax], [corr_y, min_x], [max_y, corr_x]])
        
    else: # bounding box 
        
        lower_y = all_y[all_x <= mean_x]
        upper_y = all_y[all_x > mean_x]
        
        lower_x = np.min(all_x)
        ul_y = np.min(lower_y)
        bl_y = np.max(lower_y)

        upper_x = np.max(all_x)
        ur_y = np.min(upper_y)
        lr_y = np.max(upper_y)
        
        # plotting grid 
        needle_grid[ul_y ,lower_x] = 1 # tl
        needle_grid[bl_y, lower_x] = 1 # bl 
        needle_grid[ur_y, upper_x] = 1 # tr 
        needle_grid[lr_y, upper_x] = 1 # br
        
        # obtain array of br, tr, tl, bl 
        coords = np.array([[lr_y, upper_x], [ur_y, upper_x], [ul_y, lower_x], [bl_y, lower_x]])
    
    return coords, needle_grid 

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

def sort_lesion_size(arr):
    """
    Sorts an array in descending order while keeping track of the original indices.

    Parameters:
    arr (numpy.ndarray): The input array to be sorted.

    Returns:
    tuple: A tuple containing two numpy.ndarrays:
        - sorted_values: The sorted array in descending order.
        - sorted_indices: The corresponding indices of the original array.
    """
    
    # Get the indices that would sort the array in descending order
    sorted_indices = np.argsort(arr)[::-1]
    sorted_indices += 1 # as lesion labels are 1-indexed, instead of 0-indexed

    # Sort the array in descending order
    sorted_values = np.sort(arr)[::-1]

    return sorted_values, sorted_indices

#         # order lesion indices in order of size (largest -> smallest)
# self.tumour_sizes = self.tumour_statistics['lesion_size']
# _, self.lesion_size_idx = sort_lesion_size(self.tumour_statistics['lesion_size'])


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

    def get_rel_actions(self, needle_coords,  starting_point):
        """
        Obtains relative actions to take from starting point to points on each grid
        """
        
        # Reorderd needle coords according to distance to starting point 
        #reordered_points = needle_coords[:,closest_idx] # Reordered coords according to distance to previous point 
        
        reordered_points = needle_coords # remove ordering according to closest index 
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
        #points_idx = self.find_closest_points(self.needle_coords, closest_idx, starting_point)

        # Obtain relative actions 
        relative_actions, new_start_pos = self.get_rel_actions(self.needle_coords[closest_idx], starting_point)

        # Refine actions -> ie add firing (0 or 1) and move big actions to smaller ones (from 6 -> 2, 2, 2)
        refined_actions = self.refine_actions(relative_actions, self.needle_depths[closest_idx])
        #refined_actions = np.vstack((relative_actions, np.reshape(self.needle_depths[closest_idx], (1,-1))))
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
                
                #Loop through each x,y action 
                #If size of action is greater than 2 (ie greater than 10mm) -> split up actions to individual actions 
                if abs_indv_act > 2:
                    split_actions = sign * self.action_maps[abs_indv_act]
                else: 
                    split_actions = np.array([indv_act])
                
                paired_actions.append(split_actions)
                #paired_actions.append(np.array([indv_act]))

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

class ActionFinder_l2s():
    """
    Action finder for largest to smallest lesion 
    """
    def __init__(self, lesion_means, needle_coords, needle_depths, size_idx, start_pos = np.array([0,0])):
        
        self.lesion_means = lesion_means
        self.needle_coords = needle_coords
        self.needle_depths = needle_depths 
        self.size_idx = size_idx # array of size idx to initialise 
        self.start_pos = start_pos 
        self.action_maps = {3 : np.array([2,1]), 4 : np.array([2,2]), 5 : np.array([2,2,1]), 6 : np.array([2,2,2]), \
            7 : np.array([2,2,2,1]), 8 : np.array([2,2,2,2]), 9: np.array([2,2,2,2,1]), 10: np.array([2,2,2,2,2]), \
                11: np.array([2,2,2,2,2, 1]), 12: np.array([2,2,2,2,2,2]), 13: np.array([2,2,2,2,2,2,1])}
        NUM_LESIONS = len(lesion_means)

        self.map = {}
        for idx in range(NUM_LESIONS):
            self.map[idx] = [self.lesion_means[idx], self.needle_coords[idx]]

        #self.remaining_idx = [i for i in range(NUM_LESIONS)]
        self.remaining_idx = list(size_idx -1)
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

    def get_rel_actions(self, needle_coords,  starting_point):
        """
        Obtains relative actions to take from starting point to points on each grid
        """
        
        # Reorderd needle coords according to distance to starting point 
        #reordered_points = needle_coords[:,closest_idx] # Reordered coords according to distance to previous point 
        
        reordered_points = needle_coords # remove ordering according to closest index 
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
        #needle_means = [] # needle means of remaining lesions to be visited
        idx_map = {} # maps remaining lesion index to actual needle index
        
        for map_idx, needle_idx in enumerate(self.remaining_idx): 
            #needle_means.append(self.lesion_means[needle_idx])
            idx_map[map_idx] = needle_idx # Maps [0 1 2] -> [2 0 1] where needle_idx is actual idx of positions of lesions 
        
        # Closest idx is first one in remaining idx : indexes are ordered in order of lesion size already!
        closest_idx = self.remaining_idx[0]
        print(f"closest idx : {closest_idx}")
        
        
        # Get closest lesion 
        #closest_idx, lesion_mean = self.find_closest_lesion(needle_means, idx_map, starting_point)
        #print(f'Closest_idx : {closest_idx}')
        #print(f'Starting point {starting_point}')
        #print(f'Needle coords : {needle_coords[closest_idx]}')

        # Sample 4 needle points belonging to closest lesion. Point idx is order of close-ness
        #points_idx = self.find_closest_points(self.needle_coords, closest_idx, starting_point)

        # Obtain relative actions 
        relative_actions, new_start_pos = self.get_rel_actions(self.needle_coords[closest_idx], starting_point)

        # Refine actions -> ie add firing (0 or 1) and move big actions to smaller ones (from 6 -> 2, 2, 2)
        refined_actions = self.refine_actions(relative_actions, self.needle_depths[closest_idx])
        #refined_actions = np.vstack((relative_actions, np.reshape(self.needle_depths[closest_idx], (1,-1))))
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
                
                #Loop through each x,y action 
                #If size of action is greater than 2 (ie greater than 10mm) -> split up actions to individual actions 
                if abs_indv_act > 2:
                    split_actions = sign * self.action_maps[abs_indv_act]
                else: 
                    split_actions = np.array([indv_act])
                
                paired_actions.append(split_actions)
                #paired_actions.append(np.array([indv_act]))

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
        
        Returns:
        ----------------
        img_overlay : overlay of grid with lesions 
        grid: 
        grid_single_val: 1 where grid point occurs, 0 otherwise 
        """
        prostate_proj = self.get_projection(prostate_mask)
        lesion_proj = self.get_projection(lesion_mask)

        grid, grid_single_val = self.obtain_grid(prostate_centre)

        # Arbritrary values 5 and 10 chosen to have different scale for lesion, grid 
        img_overlay = prostate_proj*2 + lesion_proj*5 #+ grid*10 

        return img_overlay, grid, grid_single_val 

    def find_closest_point(self, target_point, points):
        """
        Find the closest coordinate point from an array of coordinate points.

        Parameters:
        - target_point (tuple): The target point for which to find the closest point.
        - points (list of tuples): The array of coordinate points.

        Returns:
        - tuple: The closest coordinate point to the target point.
        """

        target_point = np.array(target_point)
        points = np.array(points)

        distances = np.linalg.norm(points - target_point, axis=1)
        closest_index = np.argmin(distances)
        closest_point = points[closest_index]

        return tuple(closest_point)

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

    def get_wacky_labels(self, prostate_mask, tumour_centroids, tumour_vol, grid, num_needles = 3, strategy = 'box'):
        
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
        #num_needles = 5
        prostate_mask_depths = np.where(prostate_mask !=0)[-1]
        prostate_mid_point = (np.max(prostate_mask_depths) + np.min(prostate_mask_depths)) / 2

        grid_coords = np.array(np.where(grid != 0)) #grid_coords where non-zero occurs 
        num_lesions = len(tumour_centroids)
        
        if num_lesions == 0: 
            print('Chicken')
        
        fired_points = np.zeros((num_lesions, num_needles))
        fired_depths = np.zeros((num_lesions, num_needles))
        all_fired_coords = np.zeros((num_lesions, num_needles, 2))
        # firing_grid with 2 points 
        simple_grid = np.zeros((13,13,2)) # first is apex, second is base 
        
        depth = [] 
        for idx, lesion_centre in enumerate((tumour_centroids)):
            
            print(f"idx {idx}")
            
            centre_point = np.reshape(lesion_centre[0:2], (2,-1))
            lesion_vol = (tumour_vol == (idx+1))
            ##plt.figure()
            #plt.imshow(np.max(lesion_vol, axis =2))
            
            # for each lesion, obtain wacky grid points 
            coords, grid_lesion = get_wacky_points(lesion_vol, strategy = strategy)
            # add centre point to start of array 
            #plt.figure()
            #plt.imshow(grid_lesion*10 + np.max(lesion_vol, axis =2))
            centre_coords = np.array([centre_point[1], centre_point[0]])
            coords_new = np.insert(coords, 0, centre_coords.reshape(2,), axis = 0) 
            #print(f"coords {coords}")
            # Compute distance from centre of lesion to grid coords 
            #dif_to_centroid = grid_coords - centre_point
            #dist_to_centroid = np.linalg.norm(dif_to_centroid, axis = 0)
            
            # for each wacky grid point, obtain closest grid position idx 
            
            
            for coord_idx, coord in enumerate(coords_new):
                fired_points[idx, coord_idx], fired_coords = find_closest_point(coord, grid_coords)
                #print(f"idx : {idx}")
                all_fired_coords[idx, coord_idx, :] = copy.deepcopy(fired_coords)
                #print(f"{fired_points[idx, coord_idx]}")
                #print(f"All fired coords : {all_fired_coords[idx, :,:]}")                
            # coord grid positions should give index! 
            
            # Rule : Choose n closest to the centroid of lesion 
            #closest_points = np.argsort(dist_to_centroid)
            #fired_points[idx, :] = closest_points[0:num_needles]
            
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
        
        return fired_points, fired_grid, fired_grid_depth, small_grid, small_grid_depth, simple_grid, fired_depths, all_fired_coords


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
            fired_grid[coords[0] - 1 :coords[0] +1 , coords[1] -1 : coords[1] +1] = 1
            fired_grid_depth[coords[0] - 1 :coords[0] +1 , coords[1] -1 : coords[1] +1, depth] = 1

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

    def get_wacky_points(self, lesion_vol, strategy = 'edges'):
        
        """
        A function that obtains 4 grid grid points corresponding to a wacky strategy 
        
        Parameters:
        ----------
        lesion_vol(ndarray) : 200 x 200 x 96 binary mask of lesion volume 
        strategy (str) : indicates which type of points to obtain : edges or box 
        
        Returns:
        ----------
        grid_points : 4 points (4 x 2) corresponding to BR, UR, UL, BL
        
        """
        # lesion 2d projection in xy plane 
        lesion_proj = np.max(lesion_vol, axis = 2)
        all_y, all_x = np.where(lesion_proj)
        mean_x = np.mean(all_x)
        needle_grid = np.zeros_like(lesion_proj)
        
        if strategy == 'edges': 
            
            #tl
            min_x = np.min(all_x)
            corr_y = np.max(all_y[all_x == min_x])
            
            #bl
            max_y = np.max(all_y[all_x < mean_x])
            corr_x = np.min(all_x[all_y == max_y]) # left most 
            
            #br
            max_x = np.max(all_x)
            corr_ymax = np.max(all_y[all_x == max_x])
            
            #tr 
            min_y = np.min(all_y[all_x >= mean_x])
            corr_xmax = np.max(all_x[all_y == min_y])
        
            needle_grid[max_y ,corr_x] = 1 #bl
            needle_grid[corr_y, min_x] = 1 # tl 
            needle_grid[corr_ymax, max_x] = 1 # br 
            needle_grid[min_y, corr_xmax] = 1 # tr
            
            # obtain array of br, tr, tl, bl 
            coords = np.array([[corr_ymax, max_x], [min_y, corr_xmax], [corr_y, min_x], [max_y, corr_x]])
            
        else: # bounding box 
            
            lower_y = all_y[all_x <= mean_x]
            upper_y = all_y[all_x > mean_x]
            
            lower_x = np.min(all_x)
            ul_y = np.min(lower_y)
            bl_y = np.max(lower_y)

            upper_x = np.max(all_x)
            ur_y = np.min(upper_y)
            lr_y = np.max(upper_y)
            
            # plotting grid 
            needle_grid[ul_y ,lower_x] = 1 # tl
            needle_grid[bl_y, lower_x] = 1 # bl 
            needle_grid[ur_y, upper_x] = 1 # tr 
            needle_grid[lr_y, upper_x] = 1 # br
            
            # obtain array of br, tr, tl, bl 
            coords = np.array([[lr_y, upper_x], [ur_y, upper_x], [ul_y, lower_x], [bl_y, lower_x]])
        
        return coords, needle_grid 

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
        
        if (self.step == 'c2l') or (self.step == 'n2n'): #c2l or n2n 

            # sample lesion 
            lesion_idx = np.random.choice((np.arange(1, NUM_LESIONS)))
            #lesion_mask = separate_masks(multiple_masks, lesion_idx)
            lesion_mask = 1*(np.array(multiple_masks) == lesion_idx)[::2,::2,::4]
            
            # resample if empty
            empty_mask = (len(np.unique(lesion_mask)) == 1)
            
            while empty_mask:
                lesion_idx = np.random.choice((np.arange(1, NUM_LESIONS)))
                #lesion_mask = separate_masks(multiple_masks, lesion_idx)
                lesion_mask = 1*(np.array(multiple_masks) == lesion_idx)[::2,::2,::4]

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
            
            while empty_mask:
                lesion_idx = np.random.choice((np.arange(1, NUM_LESIONS)))
                #lesion_mask = separate_masks(multiple_masks, lesion_idx)
                lesion_mask = 1*(np.array(multiple_masks) == lesion_idx)[::2,::2,::4]

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
            
            inal_action = torch.tensor(self._normalise_actions(final_action))#, axis = 0)

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

            ## Obtain observation of needle volumes: 
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
            print(f"random_idx {random_idx}")
            idx = 0 
            for i in range((random_idx+1 - self.T), random_idx +1):
                print(f" image idx : {i} idx : {idx} ")
                print(f"all_pos {all_pos[i, :]}")
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
                print(f" image idx : {i} idx : {idx} ")
                needle_stack[idx, :,:,:] = torch.tensor(self.vol_creater.create_needle_vol(all_pos[i, :], max_depth))
                idx += 1
            
            #needle_vol_t_2 = self.vol_creater.create_needle_vol(all_pos[random_idx-2, :], max_depth)
            #needle_vol_t_1 = self.vol_creater.create_needle_vol(all_pos[random_idx-1, :], max_depth)
            #needle_vol_t0 = self.vol_creater.create_needle_vol(all_pos[random_idx, :], max_depth)
            #needle_stack = torch.tensor([needle_vol_t_2, needle_vol_t_1, needle_vol_t0])
        
        return needle_stack 
       
       