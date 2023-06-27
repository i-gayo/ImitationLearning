

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
        self.guide_locations = [world.gland]
        self.sampling_loc_idx = []
        self.samples = []
    
    def update(self, world):
        self.sample = unfun_interpolate(world.target, self.guide_locations[self.sampling_loc_idx])


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
