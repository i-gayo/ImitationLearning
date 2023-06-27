

import torch



## main env classes
class TPBEnv():
    #TODO: Base class for both BX and FX environments
    '''
    Transperineal biopsy (TPB) environment
    :param voxdims: float pytorch tensor [N, 3].
    :param gland:   boolean 5D pytorch tensor [N, 1, D, H, W].
    :param target:  boolean 5D pytorch tensor [N, 1, D, H, W].
    :return: the base env
    '''
    def __init__(self, **kwargs):        
        #TODO: config options to include other examples 
        self.world = LabelledImageWorld(**kwargs) #Example
        self.action = NeedleGuideSampling(self.world) #Example
        self.transition = RandomDeformationTransition(self.world) #Example
        self.observation = UltrasoundSlicing(self.world) #Example

    def generate_episodes(self, num):
        for step in range(num):
            self.action.update(self.world)
            self.transition.next(self.world,self.action)
            self.observation.update(self.world)
            # assemble observations and actions
            episodes = self.transition.next()
        return episodes


## world classes
class LabelledImageWorld():
    #TODO: Base class for other world data
    def __init__(self, gland: torch.Tensor, target: torch.Tensor, voxdims: torch.Tensor):        
        #TODO: config options to include other examples 
        self.gland = gland 
        self.target = target
        self.voxdims = voxdims

        # - The normalised image coordinate system, per torch convention [-1, 1]
        # - The physical image coordinate system, in mm centred at the image centre
        #           such that coordinates_mm = coordinates_normliased * voxdims_unit (mm/unit)
        # align_corners=True, i.e. -1 and 1 are the centre points of the corner pixels (vs. corner/edge points)
        self.image_length_mm = self.gland.shape[2:5][None] * self.voxdims
        self.voxdims_unit = self.voxdims * 2 / self.image_length_mm
        self.image_centre_mm = self.image_length_mm * 0.5

        # reference_grid_*: (N, D, H, W, 3)
        self.reference_grid_normalised = torch.meshgrid()  
        self.reference_grid_mm = torch.meshgrid()  # align_corners=True

    def get_gland(self):
        return self.gland
    def get_target(self):
        return self.target


## action sampling classes
class NeedleGuideSampling():
    #TODO: base class for other sampling methods
    '''
    A class to specify needle sampling methods using a needle guide 
    Implemented here is the brachytherapy template with 13x13 5mm locations
    '''
    def __init__(self, world):
        '''
        Initialise the needle guide position, 
        aligning the gland bounding box and template centres
        '''
        self.guide_locations = self.centre_aligned_locations(world.gland, world.voxdims)
        self.sampling_loc_idx = []
        self.samples = []
    
    def update(self, world):
        self.sample = unfun_interpolate(world.target, self.guide_locations[self.sampling_loc_idx])
    
    @staticmethod
    def centre_aligned_locations(gland, voxdims):



## transition classes
class RandomDeformationTransition():
    #TODO: base class for other transition classes
    '''
    A class for world data (here, gland and target) transition
    '''
    def __init__(self):
        '''
        A transition function
        Random - no action affected the world transition
        '''

    def __next__(self, world, action):
        self.gland, self.target = unfun_deform(world.gland, world.target)


## observation classes
class UltrasoundSlicing():
    #TODO: base class for other types of observation
    '''
    A class to acquire ultrasound slices 
    '''
    def __init__(self):
        '''
        Configure the slices required for observation
        Precompute the 
        '''
    def update(self, world):
        ultrasound_slices = unfun_slicing(world.gland, world.target)
