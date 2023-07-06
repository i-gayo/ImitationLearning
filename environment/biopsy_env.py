import torch


## main env classes
class TPBEnv:
    # TODO: Base class for both BX and FX environments
    """
    Transperineal biopsy (TPB) environment
    :param voxdims: float pytorch tensor [N, 3].
    :param gland:   boolean 5D pytorch tensor [N, 1, D, H, W].
    :param target:  boolean 5D pytorch tensor [N, 1, D, H, W].
    :return: the base env
    """

    def __init__(self, **kwargs):

        MAX_STEPS = 100

        self.max_steps = MAX_STEPS
        # TODO: config options to include other examples
        self.world = LabelledImageWorld(**kwargs)  # Example
        self.action = NeedleGuide(self.world)  # Example
        self.transition = DeformationTransition(self.world)  # Example
        self.observation = UltrasoundSlicing(self.world)  # Example

    def run(self):
        episodes = []
        for step in range(self.max_steps):
            self.action.update(self.world, self.observation)
            self.transition.update(self.world, self.action)
            self.observation.update(self.world)
            # assemble observations and actions
            episodes[step] = self.transition.next()
        return episodes


## world classes
class LabelledImageWorld:
    # TODO: Base class for other world data
    def __init__(self, gland: torch.Tensor, target: torch.Tensor, voxdims: list):
        INITIAL_OBSERVE_LOCATION = "random"  # ("random", "centre")
        INITIAL_OBSERVE_RANGE = 0.3

        self.device = gland.device
        # TODO: config options to include other examples
        self.gland = gland
        self.target = target
        self.voxdims = voxdims
        self.batch_size = self.gland.shape[0]
        self.vol_size = list(self.gland.shape[2:5])

        # - The normalised image coordinate system, per torch convention [-1, 1]
        # - The physical image coordinate system, in mm centred at the image centre
        #           such that coordinates_mm = coordinates_normliased * unit_dims (mm/unit)
        # align_corners=True, i.e. -1 and 1 are the centre points of the corner pixels (vs. corner/edge points)
        self.vol_size_mm = [
            [(s - 1) * vd for s, vd in zip(self.vol_size, n)] for n in self.voxdims
        ]
        self.unit_dims = torch.tensor(
            [[u / 2 for u in n] for n in self.vol_size_mm]
        ).to(self.device)

        # precompute
        self.reference_grid_mm = self.get_reference_grid_mm()

        ## initialise the observation location
        if INITIAL_OBSERVE_LOCATION == "random":  # middle block of the image
            self.observe_mm = (
                (torch.rand(self.batch_size, 3).to(self.device) - 0.5)
                * INITIAL_OBSERVE_RANGE
                * self.unit_dims
            )
        elif INITIAL_OBSERVE_LOCATION == "centre":
            self.observe_mm = torch.zeros(self.batch_size, 3).to(self.device)

    @property
    def observe_norm(self):
        return self.convert_mm2norm(self.observe_mm)

    def get_reference_grid_mm(self):
        # reference_grid_*: (N, D, H, W, 3)
        # N.B. ij indexing for logical indexing, align_corners=True
        return torch.stack(
            [
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(
                            -self.vol_size_mm[n][0] / 2,
                            self.vol_size_mm[n][0] / 2,
                            self.vol_size[0],
                        ),
                        torch.linspace(
                            -self.vol_size_mm[n][1] / 2,
                            self.vol_size_mm[n][1] / 2,
                            self.vol_size[1],
                        ),
                        torch.linspace(
                            -self.vol_size_mm[n][2] / 2,
                            self.vol_size_mm[n][2] / 2,
                            self.vol_size[2],
                        ),
                        indexing="ij",
                    ),
                    dim=3,
                )
                for n in range(self.batch_size)
            ],
            dim=0,
        ).to(self.device)

    def get_gland_coordinates(self):
        return self.reference_grid_mm[
            self.gland.squeeze(dim=1)[..., None].repeat_interleave(dim=4, repeats=3)
        ].reshape((self.batch_size, -1, 3))
        """debug
        from PIL import Image
        im = Image.fromarray((world.gland.squeeze()[...,None].repeat_interleave(dim=4,repeats=3))[0,40,:,:,0].cpu().numpy())
        im.save("test.jpeg")
        """

    def get_target_coordinates(self):
        return self.reference_grid_mm[
            self.target.squeeze(dim=1)[..., None].repeat_interleave(dim=4, repeats=3)
        ].reshape((self.batch_size, -1, 3))

    def get_reference_slice_axial(self, norm=True):
        # N.B. xy indexing for grid_sampling
        reference_slice_axial = torch.stack(
            [
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(
                            -self.vol_size_mm[n][2] / 2,
                            self.vol_size_mm[n][2] / 2,
                            self.vol_size[2],
                        ),
                        torch.linspace(
                            -self.vol_size_mm[n][1] / 2,
                            self.vol_size_mm[n][1] / 2,
                            self.vol_size[1],
                        ),
                        torch.tensor([0.0]),
                        indexing="xy",
                    ),
                    dim=3,
                )
                for n in range(self.batch_size)
            ],
            dim=0,
        ).to(self.device)
        if norm:
            reference_slice_axial = self.convert_mm2norm(reference_slice_axial)
        return reference_slice_axial

    def get_reference_slice_sagittal(self, norm=True):
        # N.B. xy indexing for grid_sampling
        reference_slice_sagittal = torch.stack(
            [
                torch.stack(
                    torch.meshgrid(
                        torch.tensor([0.0]),
                        torch.linspace(
                            -self.vol_size_mm[n][1] / 2,
                            self.vol_size_mm[n][1] / 2,
                            self.vol_size[1],
                        ),
                        torch.linspace(
                            -self.vol_size_mm[n][0] / 2,
                            self.vol_size_mm[n][0] / 2,
                            self.vol_size[0],
                        ),
                        indexing="xy",
                    ),
                    dim=3,
                )
                for n in range(self.batch_size)
            ],
            dim=0,
        ).to(self.device)
        if norm:
            reference_slice_sagittal = self.convert_mm2norm(reference_slice_sagittal)
        return reference_slice_sagittal

    def convert_mm2norm(self, input_tensor):
        r = len(input_tensor.shape)
        if r == 5:
            return input_tensor / self.unit_dims.reshape(self.batch_size, 1, 1, 1, 3)
        elif r == 2:
             return input_tensor / self.unit_dims


## action sampling classes
class NeedleGuide:
    # TODO: base class for other sampling methods
    """
    A class to specify needle sampling methods using a needle guide
    Implemented here is the brachytherapy template with 13x13 5mm locations
    """

    def __init__(self, world):
        """
        Initialise the needle guide position,
        aligning the gland bounding box and template centres
        """
        GRID_CENTRE = "centroid"  # ("centroid", "bbox")
        GRID_SIZE = [13, 13, 5]  # [x, y, spacing (mm)]
        NEEDLE_LENGTH = 10  # in mm
        NUM_NEEDLE_DEPTHS = 3  # integer, [2, needle_length]
        NUM_NEEDLE_SAMPLES = NEEDLE_LENGTH  # int(2*NEEDLE_LENGTH+1)
        POLICY = "lesion_centre"  # GUIDANCE = 'nb27' # 'nb8'
        STEPSIZE = 5

        self.grid_size = GRID_SIZE
        self.needle_length = NEEDLE_LENGTH
        self.num_needle_depths = NUM_NEEDLE_DEPTHS
        self.num_needle_samples = NUM_NEEDLE_SAMPLES
        self.policy = POLICY
        self.observe_stepsize = STEPSIZE  # in mm
        self.num_needle_samples = NUM_NEEDLE_SAMPLES

        ## align brachytherapy template (13x13 5mm apart)
        gland_coordinates = world.get_gland_coordinates()
        if GRID_CENTRE == "centroid":
            self.grid_centre = gland_coordinates.mean(dim=1)
        elif GRID_CENTRE == "bbox":
            self.grid_centre = (
                gland_coordinates.max(dim=1)[0] + gland_coordinates.min(dim=1)[0]
            ) / 2

        ## initialise sampling
        #  a list of NUM_NEEDLE_DEPTHS (batch,13,13,NUM_NEEDLE_SAMPLES,3)
        (
            self.needle_samples_mm,
            self.sample_x,
            self.sample_y,
            self.sample_d,
        ) = self.get_needle_samples_mm(world.batch_size, world.device)
        # TODO: check the target covered by the guide locations
        # convert to normalised coordinates, TODO: add any offset in mm here
        self.needle_samples_norm = [
            world.convert_mm2norm(mm) for mm in self.needle_samples_mm
        ]

        ## initialise guidance
        # observe: two-class [negative, positive]
        # TODO: add different guidance method
        self.observe_update = torch.zeros(
            world.batch_size, 3, 2, dtype=torch.bool, device=world.device
        )

    def get_needle_samples_mm(self, batch_size, device):
        needle_centre_d = torch.linspace(
            -self.needle_length / 2, self.needle_length / 2, self.num_needle_depths
        ).tolist()
        # a list of NUM_NEEDLE_DEPTHS (batch,13,13,NUM_NEEDLE_SAMPLES,3)
        needle_samples_mm = [
            torch.stack(
                [
                    torch.stack(
                        torch.meshgrid(
                            torch.linspace(
                                -(self.grid_size[0] - 1) * self.grid_size[2] / 2,
                                (self.grid_size[0] - 1) * self.grid_size[2] / 2,
                                self.grid_size[0],
                            ),
                            torch.linspace(
                                -(self.grid_size[1] - 1) * self.grid_size[2] / 2,
                                (self.grid_size[1] - 1) * self.grid_size[2] / 2,
                                self.grid_size[1],
                            ),
                            torch.linspace(
                                centre_d - self.needle_length / 2,
                                centre_d + self.needle_length / 2,
                                self.num_needle_samples,
                            ),
                            indexing="xy",
                        ),
                        dim=3,
                    ).to(device)
                    + self.grid_centre[n, (2, 1, 0)].reshape(
                        1, 1, 1, 3
                    )  # convert to (x,y,z) from (k,j,i)
                    for n in range(batch_size)
                ],
                dim=0,
            )  # for each data in a batch then stack in dim=0
            for centre_d in needle_centre_d  # note that the grid_sample function does not accept multi-channel grid input, so list is used here
        ]  # for each needle depth
        # initialise the sampling location
        """ if use an index of [n, NUM_NEEDLE_DEPTHS, y, x]
        nc = NUM_NEEDLE_DEPTHS*GRID_SIZE[0]*GRID_SIZE[1]
        if INITIAL_SAMPLE_LOCATION == 'random':
            flat_idx = torch.randint(high=nc,size=[world.batch_size])
        elif INITIAL_SAMPLE_LOCATION == 'centre':
            flat_idx = torch.tensor([int((nc-.5)/2)]*2) 
        self.sample_location_index = torch.nn.functional.one_hot(flat_idx,nc).type(torch.bool).view(
            world.batch_size, NUM_NEEDLE_DEPTHS, GRID_SIZE[1], GRID_SIZE[0]).to(device)
        """
        sample_d = (
            torch.ones(batch_size, self.num_needle_depths, device=device)
            / self.num_needle_depths
        )
        sample_x = (
            torch.ones(batch_size, self.grid_size[0], device=device) / self.grid_size[0]
        )
        sample_y = (
            torch.ones(batch_size, self.grid_size[1], device=device) / self.grid_size[1]
        )
        return (needle_samples_mm, sample_x, sample_y, sample_d)

    def update(self, world, observation):
        ## calculate the action according to a policy
        if self.policy == "lesion_centre":
            """
            Implement "lesion_centre" policy:
             - check observe_mm nearest to the lesion centre,
             - if false, update observe_update
             - if true, update sample_x, y, z, with largest CCL
            """

            target_coordinates = world.get_target_coordinates()
            self.target_centre = target_coordinates.mean(dim=1)  # current lesion centre

            # if three_class: observe_update = ((self.target_centre-world.observe_mm) / (self.observe_stepsize*2)).round().sign()
            # self.observe_update = torch.nn.functional.one_hot((observe_update+1).type(torch.int64),num_classes=3)
            d_t2o = (
                self.target_centre - world.observe_mm
            )  # difference between current observe location and lesion centre
            self.observe_update = torch.stack(
                (d_t2o >= self.observe_stepsize, d_t2o <= -self.observe_stepsize), dim=2
            )
            if self.observe_update.any():  # update observe_update
                self.observe_update_mm = (
                    self.observe_update[..., 0] * self.observe_stepsize
                    - self.observe_update[..., 1] * self.observe_stepsize
                )
                world.observe_mm += self.observe_update_mm

            else:  # update sampled needles (batch,num_needle_depths,grid_size,grid_size,num_needle_samples)
                needle_sampled = torch.concat(
                    [
                        self.sampler(
                            world.target.type(torch.float32),
                            self.needle_samples_norm[d],
                        )
                        for d in range(self.num_needle_depths)
                    ],
                    dim=1,
                )
                core_length = needle_sampled.sum(dim=4)

            return 0

    def sample_needles(self):
        return 0

    @staticmethod
    # N.B. grid_sample uses image convention:
    #   return an interpolated volume in (y,x,z) order, here (j,i,k) or (h,w,d)
    # the input grid (...,3) should be in (x,y,z) coordinates
    # whilst tensor convention: in (k,j,i) or (d,h,w)
    def sampler(vol, coords):
        return torch.nn.functional.grid_sample(
            input=vol,
            grid=coords,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


## transition classes
class DeformationTransition:
    # TODO: base class for other transition classes
    """
    A class for world data (here, gland and target) transition
    """

    def __init__(self, world):
        """
        A transition function
        Random - no action affected the world transition
        """
        FFD_GRID_SIZE = 50
        """
        self.ffd_ctrl_pts = torch.meshgrid(
            torch.linspace(),
            torch.linspace(),
            torch.linspace(),
        )
        """

    def update(self, world, action):
        ## update the world
        # TODO: deform the volume
        return 0


## observation classes
class UltrasoundSlicing:
    # TODO: base class for other types of observation
    """
    A class to acquire ultrasound slices
    """

    def __init__(self, world):
        """
        Configure the slices required for observation
        """

        ## initial a list of slices with observation locations 1 orthogonal axial and 1 sagittal slices
        # centre: at the centre of the image volume, [(n,1,200,200,3),(n,96,200,1,3)]
        # TODO: support multiple non-orthogonal slices
        self.reference_observe_slices_norm = [
            world.get_reference_slice_axial(norm=True),
            world.get_reference_slice_sagittal(norm=True),
        ]

        self.update(world)  # get initial observation

    def update(self, world):
        # transformation TODO: add rotation for non-orthogonal reslicing
        slices_norm = [
            s + world.observe_norm.reshape(world.batch_size, 1, 1, 1, 3)
            for s in self.reference_observe_slices_norm
        ]

        # interpolation
        gland_slices = [
            self.reslicer(world.gland.type(torch.float32), g) for g in slices_norm
        ]
        target_slices = [
            self.reslicer(world.target.type(torch.float32), g) for g in slices_norm
        ]
        # gather here all the observed
        self.observation = [gland_slices, target_slices]
        """debug
        import SimpleITK as sitk
        threshold = 0.45
        for b in range(world.batch_size):
            sitk.WriteImage(sitk.GetImageFromArray((world.gland[b,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_gland.nii'%b)
            sitk.WriteImage(sitk.GetImageFromArray((world.target[b,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_target.nii'%b)
            sitk.WriteImage(sitk.GetImageFromArray((self.observation[0][0][b,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_gland_axis.jpg'%b)
            sitk.WriteImage(sitk.GetImageFromArray((self.observation[0][1][b,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_gland_sag.jpg'%b)
            sitk.WriteImage(sitk.GetImageFromArray((self.observation[1][0][b,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_target_axis.jpg'%b)
            sitk.WriteImage(sitk.GetImageFromArray((self.observation[1][1][b,...].squeeze().cpu().numpy()>=threshold).astype('uint8')*255), 'b%d_target_sag.jpg'%b)
        return 0
        """

    @staticmethod
    # N.B. grid_sample uses image convention:
    #   return an interpolated volume in (y,x,z) order, here (j,i,k) or (h,w,d)
    # the input grid (...,3) should be in (x,y,z) coordinates
    # whilst tensor convention: in (k,j,i) or (d,h,w)
    def reslicer(vol, coords):
        return torch.nn.functional.grid_sample(
            input=vol,
            grid=coords,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
