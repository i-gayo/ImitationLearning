

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
        self.transition = DeformationTransition(self.world) #Example
        self.observation = UltrasoundSlicing(self.world) #Example

    def run(self, num):
        episodes = []
        for step in range(num):
            self.action.update(self.observation)
            self.transition.next(self.world, self.action)
            self.observation.update(self.world)
            # assemble observations and actions
            episodes[step] = self.transition.next()
        return episodes


## world classes
class LabelledImageWorld():
    #TODO: Base class for other world data
    def __init__(self, gland: torch.Tensor, target: torch.Tensor, voxdims: list):       

        INITIAL_OBSERVE_LOCATION = "random" # ("random", "centre")
        INITIAL_RANGE = 0.2
     
        #TODO: config options to include other examples 
        self.gland = gland 
        self.target = target
        self.voxdims = voxdims
        self.batch_size = self.gland.shape[0]
        self.vol_size = list(self.gland.shape[2:5])

        # - The normalised image coordinate system, per torch convention [-1, 1]
        # - The physical image coordinate system, in mm centred at the image centre
        #           such that coordinates_mm = coordinates_normliased * unit_dims (mm/unit)
        # align_corners=True, i.e. -1 and 1 are the centre points of the corner pixels (vs. corner/edge points)
        self.vol_size_mm = [[(s-1)*vd for s,vd in zip(self.vol_size,n)] for n in self.voxdims]
        device = self.gland.device
        self.unit_dims = torch.tensor([[u/2 for u in n] for n in self.vol_size_mm]).to(device)

        # precompute reference_grid_*: (N, D, H, W, 3)
        self.reference_grid_mm = torch.stack([torch.stack(torch.meshgrid(
            torch.linspace(-self.vol_size_mm[n][0]/2,self.vol_size_mm[n][0]/2, self.vol_size[0]),
            torch.linspace(-self.vol_size_mm[n][1]/2,self.vol_size_mm[n][1]/2, self.vol_size[1]),
            torch.linspace(-self.vol_size_mm[n][2]/2,self.vol_size_mm[n][2]/2, self.vol_size[2]),
            indexing='ij'), dim=3) for n in range(self.batch_size)], dim=0).to(device)
        # align_corners=True

        ## initialise the observation location
        if INITIAL_OBSERVE_LOCATION == 'random': # middle block of the image
            self.observe_mm = (torch.rand(self.batch_size,3).to(device)-0.5)*INITIAL_RANGE * self.unit_dims
        elif INITIAL_OBSERVE_LOCATION == 'centre':
            self.observe_mm = torch.zeros(self.batch_size,3).to(device)

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
        GLAND_CENTRE = "centroid" # ("centroid", "bbox")
        NEEDLE_LENGTH = 10  # in mm
        NEEDLE_DEPTH = 3 # integer, [2, needle_length]
        NUM_NEEDLE_SAMPLES = NEEDLE_LENGTH  # int(2*NEEDLE_LENGTH+1)
        INITIAL_SAMPLE_LOCATION = "random" # ("random", "centre")

        ## pre-compute
        gland_coordinates = world.reference_grid_mm[
            world.gland.squeeze()[...,None].repeat_interleave(dim=4,repeats=3)
                                                    ].reshape((world.batch_size,-1,3))
        '''debug
        from PIL import Image
        im = Image.fromarray((world.gland.squeeze()[...,None].repeat_interleave(dim=4,repeats=3))[0,40,:,:,0].cpu().numpy())
        im.save("test.jpeg")
        '''
        if GLAND_CENTRE == "centroid":
            self.gland_centre = gland_coordinates.mean(dim=1)
        elif GLAND_CENTRE == "bbox":
            self.gland_centre = (gland_coordinates.max(dim=1)[0] + gland_coordinates.min(dim=1)[0]) / 2
        
        ## align brachytherapy template (13x13 5mm apart) 
        #  shape: [N,NEEDLE_DEPTH,13,13,3]
        device = world.gland.device
        # compute the starting needle locations (without length), use guide_sampling to obtain all sample locations (with length)
        self.guide_start_mm = torch.stack([
            torch.stack(
                torch.meshgrid(
                    torch.linspace(-NEEDLE_LENGTH,(2-1/NEEDLE_DEPTH)*NEEDLE_LENGTH,NEEDLE_DEPTH),
                    torch.linspace(-12*5/2,12*5/2,13),
                    torch.linspace(-12*5/2,12*5/2,13),
                    indexing='ij'), dim=3
                        ).to(device) + self.gland_centre[n].reshape(1,1,1,3) for n in range(world.batch_size)
                                                ], dim=0)
        #TODO: check the target covered by the guide locations

        ## initialises the current sampling location - an index of [n, needle_depth, y, x]
        nc = self.guide_start_mm.shape[1]*self.guide_start_mm.shape[2]*self.guide_start_mm.shape[3]
        if INITIAL_SAMPLE_LOCATION == 'random':
            flat_idx = torch.randint(high=nc,size=[world.batch_size])
        elif INITIAL_SAMPLE_LOCATION == 'centre':
            flat_idx = torch.tensor([int((nc-.5)/2)]*2)        
        self.sample_location_index = torch.nn.functional.one_hot(flat_idx,nc).type(torch.bool).view(
            world.batch_size, 
            NEEDLE_DEPTH, 
            self.guide_start_mm.shape[2], 
            self.guide_start_mm.shape[3]
            ).to(device)


    def update(self, observation):
        #self.sample = unfun_interpolate(world.target, self.guide_locations[self.sampling_loc_idx])
        return 0


## transition classes
class DeformationTransition():
    #TODO: base class for other transition classes
    '''
    A class for world data (here, gland and target) transition
    '''
    def __init__(self, world):
        '''
        A transition function
        Random - no action affected the world transition
        '''
        FFD_GRID_SIZE = 50
        '''
        self.ffd_ctrl_pts = torch.meshgrid(
            torch.linspace(),
            torch.linspace(),
            torch.linspace(),
        )
        '''

    def __next__(self, world, action):
        self.gland, self.target = unfun_deform(world.gland, world.target)


## observation classes
class UltrasoundSlicing():
    #TODO: base class for other types of observation
    '''
    A class to acquire ultrasound slices 
    '''
    def __init__(self, world):
        '''
        Configure the slices required for observation
        Precompute the 
        '''
        
        ## initial observation locations of 1 orthogonal axial and 1 sagittal slices
        #TODO: support multiple non-orthogonal slices
        # centre: at the centre of the image volume, [(n,1,200,200,3),(n,96,200,1,3)]
        device = world.gland.device
        self.reference_slices_mm = [
            torch.stack([torch.stack(torch.meshgrid(
            torch.tensor([.0]),
            torch.linspace(-world.vol_size_mm[n][1]/2,world.vol_size_mm[n][1]/2, world.vol_size[1]),
            torch.linspace(-world.vol_size_mm[n][2]/2,world.vol_size_mm[n][2]/2, world.vol_size[2]),
            indexing='ij'), dim=3) for n in range(world.batch_size)], dim=0).to(device),        
            torch.stack([torch.stack(torch.meshgrid(
            torch.linspace(-world.vol_size_mm[n][0]/2,world.vol_size_mm[n][0]/2, world.vol_size[0]),
            torch.linspace(-world.vol_size_mm[n][1]/2,world.vol_size_mm[n][1]/2, world.vol_size[1]),
            torch.tensor([.0]),
            indexing='ij'), dim=3) for n in range(world.batch_size)], dim=0).to(device)
        ]

        self.update(world)  # get initial observation

    def update(self, world):
        # transformation TODO: add rotation for non-orthogonal reslicing 
        slices_norm = [
            (s + world.observe_mm.reshape(world.batch_size,1,1,1,3)) 
            / world.unit_dims.reshape(world.batch_size,1,1,1,3) 
            for s in self.reference_slices_mm
            ]
        # interpolation
        gland_slices = [self.reslice(world.gland.type(torch.float32), s) for s in slices_norm]
        target_slices = [self.reslice(world.target.type(torch.float32), s) for s in slices_norm]
        # gather here all the observed 
        self.observation = [gland_slices, target_slices]
    
    @staticmethod
    def reslice(vol, coords):
        return torch.nn.functional.grid_sample(
            input = vol, 
            grid = coords, 
            mode = 'bilinear', 
            padding_mode = 'zeros',
            align_corners = True
            )

