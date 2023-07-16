import torch

from environment.utils import GridTransform


## main env classes
class TPBEnv:
    # TODO: Base class for both BX and FX environments
    """
    Transperineal biopsy (TPB) environment
    :param voxdims: float pytorch tensor (N, 3).
    :param gland:   boolean 5D pytorch tensor (N, C, Y, X, Z).
    :param target:  boolean 5D pytorch tensor (N, C, Y, X, Z).
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
            episodes += [(self.world, self.action, self.observation)]
            if self.action.sample_status.all():
                break
        return episodes


## world classes
class LabelledImageWorld:
    # TODO: Base class for other world data
    def __init__(self, gland: torch.Tensor, target: torch.Tensor, voxdims: list):
        """
        gland: (y,x,z) volume
        target: (y,x,z) volume
        voxdims: (x,y,z) voxel dimensions mm/unit
        """
        INITIAL_OBSERVE_LOCATION = "random"  # ("random", "centre")
        INITIAL_OBSERVE_RANGE = 0.3

        self.device = gland.device
        # TODO: config options to include other examples
        self.gland = gland
        self.target = target
        self.voxdims = voxdims  # x-y-z order
        self.batch_size = self.gland.shape[0]  # x-y-z order
        self.vol_size = [self.gland.shape[i] for i in [3, 2, 4]]

        # - The normalised image coordinate system, per torch convention [-1, 1]
        # - The physical image coordinate system, in mm centred at the image centre
        #           such that coordinates_mm = coordinates_normliased * unit_dims (mm/unit), x-y-z order
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
                            -self.vol_size_mm[n][1] / 2,
                            self.vol_size_mm[n][1] / 2,
                            self.vol_size[1],
                        ),
                        torch.linspace(
                            -self.vol_size_mm[n][0] / 2,
                            self.vol_size_mm[n][0] / 2,
                            self.vol_size[0],
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

    def get_mask_coords_mm(self, mask):
        # mask: (b,1,y,x,z)
        # return a list of coordinates
        return [
            self.reference_grid_mm[b, mask.squeeze(dim=1)[b, ...], :]
            for b in range(self.batch_size)
        ]

    def get_reference_slice_axial(self, norm=True):
        reference_slice_axial = torch.stack(
            [
                torch.stack(
                    torch.meshgrid(
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
                        torch.tensor([0.0]),
                        indexing="ij",
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
        reference_slice_sagittal = torch.stack(
            [
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(
                            -self.vol_size_mm[n][1] / 2,
                            self.vol_size_mm[n][1] / 2,
                            self.vol_size[1],
                        ),
                        torch.tensor([0.0]),
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
        if norm:
            reference_slice_sagittal = self.convert_mm2norm(reference_slice_sagittal)
        return reference_slice_sagittal

    def convert_mm2norm(self, input_tensor):
        r = len(input_tensor.shape)
        if r == 5:
            return input_tensor / self.unit_dims.view(self.batch_size, 1, 1, 1, 3)
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
        NEEDLE_LENGTH = 20  # in mm
        NUM_NEEDLE_DEPTHS = 3  # integer, [2, needle_length]
        NUM_NEEDLE_SAMPLES = NEEDLE_LENGTH  # int(2*NEEDLE_LENGTH+1)
        OBSERVE_POLICY = "lesion_centre"  # GUIDANCE = 'nb27' # 'nb8'
        SAMPLE_POLICY = "mccl"  # "mccl"  # "lesion_centre"
        STEPSIZE = 5

        self.grid_size = GRID_SIZE
        self.needle_length = NEEDLE_LENGTH
        self.num_needle_depths = NUM_NEEDLE_DEPTHS
        self.num_needle_samples = NUM_NEEDLE_SAMPLES
        self.observe_policy = OBSERVE_POLICY
        self.sample_policy = SAMPLE_POLICY
        self.observe_stepsize = STEPSIZE  # in mm
        self.num_needle_samples = NUM_NEEDLE_SAMPLES

        ## align brachytherapy template (13x13 5mm apart)
        gland_coords_mm = world.get_mask_coords_mm(world.gland)
        if GRID_CENTRE == "centroid":
            self.grid_centre = torch.stack(
                [gland_coords_mm[b].mean(dim=0) for b in range(world.batch_size)],
                dim=0,
            )
        elif GRID_CENTRE == "bbox":
            self.grid_centre = torch.stack(
                [
                    (
                        gland_coords_mm[b].max(dim=0)[0]
                        + gland_coords_mm[b].min(dim=0)[0]
                    )
                    / 2
                    for b in range(world.batch_size)
                ],
                dim=0,
            )

        ## initialise sampling
        #  a list of NUM_NEEDLE_DEPTHS (batch,13,13,NUM_NEEDLE_SAMPLES,3)
        (
            self.needle_samples_mm,
            self.needle_centres_mm,
        ) = self.get_needle_samples_mm(world.batch_size, world.device)
        # TODO: check the target covered by the guide locations
        # convert to normalised coordinates, TODO: add any offset in mm here
        self.needle_samples_norm = [
            world.convert_mm2norm(mm) for mm in self.needle_samples_mm
        ]
        # sample_x/y 13-class classification, sample_d NUM_NEEDLE_DEPTHS-class classification
        self.sample_x, self.sample_y, self.sample_d = (
            torch.zeros(
                world.batch_size,
                self.grid_size[0],
                dtype=torch.bool,
                device=world.device,
            ),
            torch.zeros(
                world.batch_size,
                self.grid_size[1],
                dtype=torch.bool,
                device=world.device,
            ),
            torch.zeros(
                world.batch_size,
                self.num_needle_depths,
                dtype=torch.bool,
                device=world.device,
            ),
        )
        """ if use an index of [n, NUM_NEEDLE_DEPTHS, y, x] - a 13*13*NUM_NEEDLE_DEPTHS classification
        nc = NUM_NEEDLE_DEPTHS*GRID_SIZE[0]*GRID_SIZE[1]
        if INITIAL_SAMPLE_LOCATION == 'random':
            flat_idx = torch.randint(high=nc,size=[world.batch_size])
        elif INITIAL_SAMPLE_LOCATION == 'centre':
            flat_idx = torch.tensor([int((nc-.5)/2)]*2) 
        self.sample_location_index = torch.nn.functional.one_hot(flat_idx,nc).type(torch.bool).view(
            world.batch_size, NUM_NEEDLE_DEPTHS, GRID_SIZE[1], GRID_SIZE[0]).to(device)
        """

        ## initialise guidance
        # observe: [x,y,z] * [negative, positive] binary classifications
        # TODO: add different guidance method
        self.observe_update = torch.zeros(
            world.batch_size, 3, 2, dtype=torch.bool, device=world.device
        )

    def get_needle_samples_mm(self, batch_size, device):
        needle_centre_d = torch.linspace(
            -self.needle_length / 2, self.needle_length / 2, self.num_needle_depths
        ).tolist()
        # a list of NUM_NEEDLE_DEPTHS (batch,13,13,NUM_NEEDLE_SAMPLES,yxz)
        needle_samples_mm = [
            torch.stack(
                [
                    torch.stack(
                        torch.meshgrid(
                            torch.linspace(
                                -(self.grid_size[1] - 1) * self.grid_size[2] / 2,
                                (self.grid_size[1] - 1) * self.grid_size[2] / 2,
                                self.grid_size[1],
                            ),
                            torch.linspace(
                                -(self.grid_size[0] - 1) * self.grid_size[2] / 2,
                                (self.grid_size[0] - 1) * self.grid_size[2] / 2,
                                self.grid_size[0],
                            ),
                            torch.linspace(
                                centre_d - self.needle_length / 2,
                                centre_d + self.needle_length / 2,
                                self.num_needle_samples,
                            ),
                            indexing="ij",
                        ),
                        dim=3,
                    ).to(device)
                    + self.grid_centre[n].view(1, 1, 1, 3)
                    for n in range(batch_size)
                ],
                dim=0,
            )  # for each data in a batch then stack in dim=0
            for centre_d in needle_centre_d  # note that the grid_sample function does not accept multi-channel grid input, so list is used here
        ]  # for each needle depth

        # a single array (batch,13,13,NUM_NEEDLE_DEPTHS,yxz)
        needle_centres_mm = torch.concat(
            [
                torch.stack(
                    [
                        torch.stack(
                            torch.meshgrid(
                                torch.linspace(
                                    -(self.grid_size[1] - 1) * self.grid_size[2] / 2,
                                    (self.grid_size[1] - 1) * self.grid_size[2] / 2,
                                    self.grid_size[1],
                                ),
                                torch.linspace(
                                    -(self.grid_size[0] - 1) * self.grid_size[2] / 2,
                                    (self.grid_size[0] - 1) * self.grid_size[2] / 2,
                                    self.grid_size[0],
                                ),
                                torch.tensor(centre_d),
                                indexing="ij",
                            ),
                            dim=3,
                        ).to(device)
                        + self.grid_centre[n].view(1, 1, 1, 3)
                        for n in range(batch_size)
                    ],
                    dim=0,
                )  # for each data in a batch then stack in dim=0
                for centre_d in needle_centre_d  # note that the grid_sample function does not accept multi-channel grid input, so list is used here
            ],  # for each needle depth
            dim=3,
        )

        return (needle_samples_mm, needle_centres_mm)

    def update(self, world, observation):
        """
        Calculate the action according to a policy
         - update observe_mm, if no updates, return sample_status = True
         - when all True, update sampled needles
        """
        if self.observe_policy == "lesion_centre":
            """
            Implement "lesion_centre" policy:
             - check observe_mm nearest to the lesion centre,
             - if false, update observe_update
             - if true, update sample_x, y, z, with nearest centre (or largest CCL)
            """
            # update in batch for efficiency
            target_coords_mm = world.get_mask_coords_mm(world.target)
            self.target_centre_mm = torch.stack(
                [target_coords_mm[b].mean(dim=0) for b in range(world.batch_size)],
                dim=0,
            )
            # if three_class: observe_update = ((self.target_centre-world.observe_mm) / (self.observe_stepsize*2)).round().sign()
            # self.observe_update = torch.nn.functional.one_hot((observe_update+1).type(torch.int64),num_classes=3)
            d_t2o = (
                self.target_centre_mm - world.observe_mm
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
            self.sample_status = (
                (self.observe_update == False).view(world.batch_size, -1).all(dim=1)
            )

            # update sampled needles in batch for efficiency
            # TODO: check visually
            if (
                self.observe_update == False
            ).all():  # wait until the whole batch find observe
                if self.sample_policy == "mccl":
                    # (batch,num_needle_depths,grid_size,grid_size,num_needle_samples)
                    needle_sampled_all = torch.concat(
                        [
                            self.sampler(
                                world.target.type(torch.float32),
                                self.needle_samples_norm[d],
                            )
                            for d in range(self.num_needle_depths)
                        ],
                        dim=1,
                    )
                    ccl_all = needle_sampled_all.sum(dim=4)
                    self.ccl_sampled, needle_sampled_idx_flat = ccl_all.view(
                        world.batch_size, -1
                    ).max(dim=1)
                    needle_sampled_idx = (
                        ccl_all == self.ccl_sampled.view(world.batch_size, 1, 1, 1)
                    ).nonzero()
                    self.sample_x[
                        needle_sampled_idx[:, 0], needle_sampled_idx[:, 3]
                    ] = True
                    self.sample_y[
                        needle_sampled_idx[:, 0], needle_sampled_idx[:, 2]
                    ] = True
                    self.sample_d[
                        needle_sampled_idx[:, 0], needle_sampled_idx[:, 1]
                    ] = True

                elif self.sample_policy == "lesion-centre":
                    print(
                        "WARNING: Sample at lesion centre may result in empty cores more often."
                    )
                    # the closest target to needle centre distance
                    d_t2n = (
                        (
                            (
                                self.target_centre_mm.view(world.batch_size, 1, 1, 1, 3)
                                - self.needle_centres_mm
                            )
                            ** 2
                        )
                        .sum(dim=4)
                        .sqrt()
                    )
                    d_t2n_min, needle_sampled_idx_flat = torch.min(
                        d_t2n.view(world.batch_size, -1), dim=1
                    )

                    if (d_t2n_min[0:2] >= self.grid_size[2]).any() or (
                        d_t2n_min[2] >= (self.needle_length / 2)
                    ):
                        print(
                            "Current sampled needle location may not be closet to the target."
                        )

                    needle_coords_norm = (
                        torch.stack(self.needle_samples_norm, dim=3)
                        .view(world.batch_size, -1, self.num_needle_samples, 3)[
                            range(world.batch_size), needle_sampled_idx_flat, :, :
                        ]
                        .view(world.batch_size, 1, 1, self.num_needle_samples, 3)
                    )
                    needle_sampled = self.sampler(
                        world.target.type(torch.float32), needle_coords_norm
                    )
                    self.ccl_sampled = needle_sampled.squeeze().sum(dim=1)
                    needle_sampled_idx = (
                        d_t2n == d_t2n_min.view(world.batch_size, 1, 1, 1)
                    ).nonzero()
                    self.sample_x[
                        needle_sampled_idx[:, 0], needle_sampled_idx[:, 2]
                    ] = True
                    self.sample_y[
                        needle_sampled_idx[:, 0], needle_sampled_idx[:, 1]
                    ] = True
                    self.sample_y[
                        needle_sampled_idx[:, 0], needle_sampled_idx[:, 3]
                    ] = True

    @staticmethod
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
        Random deformation only, i.e. no action affected the world transition
        """
        FFD_GRID_SIZE = [8, 8, 4]

        self.ffd_grid_size = FFD_GRID_SIZE
        self.random_transform = GridTransform(
            grid_size=self.ffd_grid_size,
            interp_type="linear",
            volsize=[world.gland.shape[i] for i in [3, 2, 4]],
            batch_size=world.batch_size,
            device=world.device,
        )

    def update(self, world, action, threshold=0.45):
        self.random_transform.generate_random_transform(rate=0, scale=0.0)
        transformed_volume = (
            self.random_transform.warp(
                torch.concat([world.gland, world.target], dim=1).type(torch.float32)
            )
            >= threshold
        )
        world.gland, world.target = (
            transformed_volume[:, [0], ...],
            transformed_volume[:, [1], ...],
        )


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
            s + world.observe_norm.view(world.batch_size, 1, 1, 1, 3)
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
    def reslicer(vol, coords):
        return torch.nn.functional.grid_sample(
            input=vol,
            grid=coords,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
