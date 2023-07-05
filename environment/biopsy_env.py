

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
        self.action = NeedleGuide(self.world) #Example
        self.transition = DeformationTransition(self.world) #Example
        self.observation = UltrasoundSlicing(self.world) #Example

    def run(self, num):
        episodes = []
        for step in range(num):
            self.action.update(self.world, self.observation)
            self.transition.update(self.world, self.action)
            self.observation.update(self.world)
            # assemble observations and actions
            episodes[step] = self.transition.next()
        return episodes


## world classes
class LabelledImageWorld():
    #TODO: Base class for other world data
    def __init__(self, gland: torch.Tensor, target: torch.Tensor, voxdims: list):       

        INITIAL_OBSERVE_LOCATION = "random" # ("random", "centre")
        INITIAL_OBSERVE_RANGE = 0.2
     
        device = gland.device
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
        self.unit_dims = torch.tensor([[u/2 for u in n] for n in self.vol_size_mm]).to(device)

        # precompute reference_grid_*: (N, D, H, W, 3)
        #N.B. ij indexing for logical indexing
        self.reference_grid_mm = torch.stack([torch.stack(torch.meshgrid(
            torch.linspace(-self.vol_size_mm[n][0]/2,self.vol_size_mm[n][0]/2, self.vol_size[0]),
            torch.linspace(-self.vol_size_mm[n][1]/2,self.vol_size_mm[n][1]/2, self.vol_size[1]),
            torch.linspace(-self.vol_size_mm[n][2]/2,self.vol_size_mm[n][2]/2, self.vol_size[2]),
            indexing='ij'), dim=3) for n in range(self.batch_size)], dim=0).to(device)
        # align_corners=True

        ## initialise the observation location
        if INITIAL_OBSERVE_LOCATION == 'random': # middle block of the image
            self.observe_mm = (torch.rand(self.batch_size,3).to(device)-0.5)*INITIAL_OBSERVE_RANGE * self.unit_dims
        elif INITIAL_OBSERVE_LOCATION == 'centre':
            self.observe_mm = torch.zeros(self.batch_size,3).to(device)
  
    def get_gland_coordinates(self):
        return self.reference_grid_mm[
            self.gland.squeeze(dim=1)[...,None].repeat_interleave(dim=4,repeats=3)
            ].reshape((self.batch_size,-1,3))
        '''debug
        from PIL import Image
        im = Image.fromarray((world.gland.squeeze()[...,None].repeat_interleave(dim=4,repeats=3))[0,40,:,:,0].cpu().numpy())
        im.save("test.jpeg")
        '''
    
    def get_target_coordinates(self):
        return self.reference_grid_mm[
            self.target.squeeze(dim=1)[...,None].repeat_interleave(dim=4,repeats=3)
            ].reshape((self.batch_size,-1,3))


## action sampling classes
class NeedleGuide():
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
        GRID_SIZE = [13, 13, 5]  # [x, y, spacing (mm)]
        NEEDLE_LENGTH = 10  # in mm
        NUM_NEEDLE_DEPTHS = 3 # integer, [2, needle_length]
        NUM_NEEDLE_SAMPLES = NEEDLE_LENGTH  # int(2*NEEDLE_LENGTH+1)
        INITIAL_SAMPLE_LOCATION = "random" # ("random", "centre")

        POLICY = 'lesion_centre'
        # GUIDANCE = 'nb27' # 'nb8'
        STEPSIZE = 5

        self.policy = POLICY
        self.observe_stepsize = STEPSIZE  # in mm
        self.num_needle_samples = NUM_NEEDLE_SAMPLES
        
        ## align brachytherapy template (13x13 5mm apart) 
        gland_coordinates = world.get_gland_coordinates()
        if GLAND_CENTRE == "centroid":
            self.gland_centre = gland_coordinates.mean(dim=1)
        elif GLAND_CENTRE == "bbox":
            self.gland_centre = (gland_coordinates.max(dim=1)[0] + gland_coordinates.min(dim=1)[0]) / 2

        # compute the needle centre locations (without length), use guide_sampling to obtain all sample locations (with length)
        #N.B. xy indexing for grid sampler
        device = world.gland.device
        needle_centre_d = torch.linspace(
            -NEEDLE_LENGTH/2,
            NEEDLE_LENGTH/2,
            NUM_NEEDLE_DEPTHS).tolist()
        # a list of NUM_NEEDLE_DEPTHS (batch,13,13,NUM_NEEDLE_SAMPLES,3)
        self.needle_samples_mm = [
            torch.stack([
            torch.stack(
                torch.meshgrid(
                    torch.linspace(
            -(GRID_SIZE[0]-1)*GRID_SIZE[2]/2,
            (GRID_SIZE[0]-1)*GRID_SIZE[2]/2,
            GRID_SIZE[0]),
                    torch.linspace(
            -(GRID_SIZE[1]-1)*GRID_SIZE[2]/2,
            (GRID_SIZE[1]-1)*GRID_SIZE[2]/2,
            GRID_SIZE[1]),
                    torch.linspace(
            centre_d - NEEDLE_LENGTH/2,
            centre_d + NEEDLE_LENGTH/2,
            NUM_NEEDLE_SAMPLES), 
                    indexing='xy'), dim=3
                        ).to(device) + self.gland_centre[n,(2,1,0)].reshape(1,1,1,3)  # convert to (x,y,z) from (k,j,i)
                        for n in range(world.batch_size)], dim=0)  # for each data in a batch then stack in dim=0
                                    for centre_d in needle_centre_d]  # for each needle depth

        #TODO: check the target covered by the guide locations

        ## initialise the sampling location 
        ''' if use an index of [n, NUM_NEEDLE_DEPTHS, y, x]
        nc = NUM_NEEDLE_DEPTHS*GRID_SIZE[0]*GRID_SIZE[1]
        if INITIAL_SAMPLE_LOCATION == 'random':
            flat_idx = torch.randint(high=nc,size=[world.batch_size])
        elif INITIAL_SAMPLE_LOCATION == 'centre':
            flat_idx = torch.tensor([int((nc-.5)/2)]*2) 
        self.sample_location_index = torch.nn.functional.one_hot(flat_idx,nc).type(torch.bool).view(
            world.batch_size, NUM_NEEDLE_DEPTHS, GRID_SIZE[1], GRID_SIZE[0]).to(device)
        '''
        self.sample_d = torch.ones(world.batch_size, NUM_NEEDLE_DEPTHS, device=device) / NUM_NEEDLE_DEPTHS
        self.sample_x = torch.ones(world.batch_size, GRID_SIZE[0], device=device) / GRID_SIZE[0]
        self.sample_y = torch.ones(world.batch_size, GRID_SIZE[1], device=device) / GRID_SIZE[1]
        
        ## initialise guidance
        # observe: [positive, zero, negative]
        #TODO: add different guidance method
        self.observe_x = torch.ones(world.batch_size, 3, device=device) / 3
        self.observe_y = torch.ones(world.batch_size, 3, device=device) / 3
        self.observe_z = torch.ones(world.batch_size, 3, device=device) / 3


    def update(self, world, observation):
        ## calculate the action according to a policy
        if self.policy == 'lesion_centre': 
            '''
            Implement "lesion_centre" policy:
             - Update the closest <sample location/depth> to the (changing due to motion) lesion centre - 13x13 classification
             - Move the <observation location> to the one closest to the lesion centre - 6 classification
             - when arrives the closest <observation location>, set optimum depth [0,0,0,1] -> [one-hot,0] - 4 classification
            '''
            ## current lesion centre
            target_coordinates = world.get_target_coordinates()
            self.target_centre = target_coordinates.mean(dim=1)

            ## find cloest observe location
            status = (self.target_centre[2]-self.observe_x) & (self.target_centre[1]-self.observe_y) & (self.target_centre[0]-self.observe_d)

            ## needle samples
            return status, self.sample_needles()
    
    def sample_needles(self):
            needle_samples = self.sampler(self.target, self.needle_samples_mm[self.sample_d][:,self.sample_x,self.sample_y,...])
    
    @staticmethod
    # N.B. grid_sample uses image convention: 
    #   return an interpolated volume in (y,x,z) order, here (j,i,k) or (h,w,d) 
    # the input grid (...,3) should be in (x,y,z) coordinates
    # whilst tensor convention: in (k,j,i) or (d,h,w)
    def sampler(vol, coords):  
        return torch.nn.functional.grid_sample(
            input = vol, 
            grid = coords, 
            mode = 'bilinear', 
            padding_mode = 'zeros',
            align_corners = True
            )


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
        

    def update(self, world, action):
        ## update the world
        #TODO: deform the volume
        return 0




## observation classes
class UltrasoundSlicing():
    #TODO: base class for other types of observation
    '''
    A class to acquire ultrasound slices 
    '''
    def __init__(self, world):
        '''
        Configure the slices required for observation
        '''
        
        ## initial observation locations of 1 orthogonal axial and 1 sagittal slices
        #TODO: support multiple non-orthogonal slices
        # centre: at the centre of the image volume, [(n,1,200,200,3),(n,96,200,1,3)]
        device = world.gland.device
        #N.B. xy indexing for grid_sampling
        self.reference_slices_mm = [
            torch.stack([torch.stack(torch.meshgrid(
            torch.linspace(-world.vol_size_mm[n][2]/2,world.vol_size_mm[n][2]/2, world.vol_size[2]),
            torch.linspace(-world.vol_size_mm[n][1]/2,world.vol_size_mm[n][1]/2, world.vol_size[1]),
            torch.tensor([.0]),
            indexing='xy'), dim=3) for n in range(world.batch_size)], dim=0).to(device),        
            torch.stack([torch.stack(torch.meshgrid(
            torch.tensor([.0]),
            torch.linspace(-world.vol_size_mm[n][1]/2,world.vol_size_mm[n][1]/2, world.vol_size[1]),
            torch.linspace(-world.vol_size_mm[n][0]/2,world.vol_size_mm[n][0]/2, world.vol_size[0]),
            indexing='xy'), dim=3) for n in range(world.batch_size)], dim=0).to(device)
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
        gland_slices = [self.reslice(world.gland.type(torch.float32), g) for g in slices_norm]
        target_slices = [self.reslice(world.target.type(torch.float32), g) for g in slices_norm]
        # gather here all the observed 
        self.observation = [gland_slices, target_slices]
        #'''debug
        import SimpleITK as sitk
        threshold = 0.45
        for b in range(world.batch_size):
            sitk.WriteImage(sitk.GetImageFromArray((world.gland[b,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_gland.nii'%b)
            sitk.WriteImage(sitk.GetImageFromArray((world.target[b,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_target.nii'%b)
            sitk.WriteImage(sitk.GetImageFromArray((self.observation[0][0][b,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_gland_axis.jpg'%b)
            sitk.WriteImage(sitk.GetImageFromArray((self.observation[0][1][b,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_gland_sag.jpg'%b)
            sitk.WriteImage(sitk.GetImageFromArray((self.observation[1][0][b,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_target_axis.jpg'%b)
            sitk.WriteImage(sitk.GetImageFromArray((self.observation[1][1][b,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_target_sag.jpg'%b)
        #'''
        return 0
    
    @staticmethod
    # N.B. grid_sample uses image convention: 
    #   return an interpolated volume in (y,x,z) order, here (j,i,k) or (h,w,d) 
    # the input grid (...,3) should be in (x,y,z) coordinates
    # whilst tensor convention: in (k,j,i) or (d,h,w)
    def reslice(vol, coords):  
        return torch.nn.functional.grid_sample(
            input = vol, 
            grid = coords, 
            mode = 'bilinear', 
            padding_mode = 'zeros',
            align_corners = True
            )
