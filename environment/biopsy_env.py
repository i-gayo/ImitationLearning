

import torch


class BaseEnv():
    '''
    Base class for both BX and FX environments
    :param gland: boolean 5D pytorch tensor [N, 1, D, H, W].
    :param target: boolean 5D pytorch tensor [N, 1, D, H, W].
    :return: the base env
    '''
    def __init__(self, gland: torch.Tensor, target: torch.Tensor):
        # world building
        self.gland = gland
        self.target = target
        # self.guide = NeedleGuide(gland) #Example
        # self.transition = RandomDeformationTransition(gland, target) #Example
    
    def generate_episodes(self, num):
        episodes = self.transition.next()
        return episodes


class TPBEnv(BaseEnv):
    '''
    Transperineal biopsy (TPB) environment
    '''
    def __init__(self):
        # world building
        self.guide = NeedleGuide(self.gland) #Example
        self.transition = RandomDeformationTransition(self.gland, self.target) #Example


## needle guiding classes
class NeedleGuide():
    #TODO: base class for other sampling methods
    '''
    A class to specify needle sampling methods using a needle guide 
    Implemented here is the brachytherapy template with 13x13 5mm locations
    '''
    def __init__(self, gland: torch.Tensor):
        '''
        Initialise the needle guide position, aligning the gland and template centres
        '''
        self.gland = gland
        self.guide_locations = []
        self.sampling_loc_idx = []
        self.samples = []
    
    def get_sample(self, target: torch.Tensor):
        self.sample = unfun_interpolate(target, self.guide_locations[self.sampling_loc_idx])


## transition classes
class RandomDeformationTransition():

    def __init__(self, gland: torch.Tensor, target: torch.Tensor):
        '''
        A transition function
        Random - no action affected the world transition
        '''
        self.gland = gland
        self.target = target

    def __next__(self):
        self.gland, self.target = unfun_deform(self.gland, self.target)
        # get_observation():
        #TODO: config file required
        ultrasound_slices = unfun_slicing(self.gland, self.target)
        yield ultrasound_slices 