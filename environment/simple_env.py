### Computes a single biopsy env based on previous env
from utils.mr2us_utils import * 
from utils.data_utils import * 
from environment.biopsy_env import * 
import time 
import h5py 

def normalise_data(img):
    """
    Normalises labels and images 
    """
    
    min_val = torch.min(img)
    max_val = torch.max(img)
    
    if max_val == 0: 
        #print(f"Empty mask, not normalised img")
        norm_img = img # return as 0s only if blank image or volume 
    else: 
        norm_img = (img - min_val) / (max_val - min_val)
    
    return norm_img 
    
    
class TargettingEnv:
    """
    A simple biopsy env that follows simple MDP:
    
    Observation:
        6 images : U_ax, U_sag, T_ax, T_sag, P_ax, P_sag
    
    Actions : 
        3 actions : 
            x : (-5,+5)
            y : (-5,+5)
            z : (1,2) # base / apex sampling
    
    Reward :
        + 10 lesion in image
        (-0.1,+) for shaped reward (getting closer to )
    """
    
    def initialise_world(self):
        # Obtains world coordinates (in mm) and image coordinates
        mr, us, gland, target, target_type, voxdims = self.data_sampler.sample_data()
        voxdims = [val.item() for val in voxdims]
        
        # Initialise world; start with centre of grid 
        self.world = LabelledImageWorld(mr,
                                        us,
                                        gland,
                                        target,
                                        voxdims) # initialises starting coordinaets 
        #self.current_pos = self.world.observe_mm
        
        # Initialise grid coordinates and positions 
        self.action = NeedleGuide(self.world)  # Example
        self.transition = DeformationTransition(self.world)  # Example
        self.observation = UltrasoundSlicing(self.world)  # Example
        
        # Initialise grid coords 
        self.apex_coords, self.base_coords = self.initialise_grid_coords()
        
        # Update current grid pos and self.world_observe_mm  : set to middle of grid 
        # Note : grid pos is in grid coords [-30,30] ; world_observe_mm is mm coord space
        self.grid_pos = torch.tensor([[0,0,0]]) # centre of prostate 
        self.world.observe_mm[:,0:2] += self.prostate_centroid[0:2].reshape(1,2)
        
        ##### Get initial observations : mid-gland, centre of grid 
        # coomment out action update as no actions yet 
        #self.action.update(self.world, self.observation)
            
        # Deform prostate and gland randomly (using parameters)
        if self.deform:
            self.transition.update(self.world, self.action)
        
        # Obtain initial obseravtions
        # consists of gland axial + sagital; target axial
        us, gland, target = self.observation.update(self.world)
        
        #us = torch.stack(us)
        initial_obs = [us,gland,target]
        # construct initial obs : stack us, gland and target together 
        
        # Testing something:
        #self.action.update(self.world, self.observation)
        
        # plt.figure()
        # plt.imshow(np.max(target.squeeze()[:,:,:].numpy(), axis = 2))
        # plt.savefig("IMGS/LESION_MASK.png")
        return initial_obs
               
    def __init__(self, data_sampler, max_steps = 20, deform = False):
        
        # Sample first patient 
        self.data_sampler = data_sampler 
        
        # Initialise counting positions for termination 
        self.step_counter = 0 
        self.max_steps = 20 
        self.deform = deform # whether to deform or not 
        
        # Obtains world coordinates (in mm) and image coordinates
        self.initialise_world()
        
    def step(self, actions = torch.tensor([1,1,1])):
        """
        Updates new observation
        Computes reward 
        
        Parameters:
        -----------
        actions: 1 x 3 array of actions chosen in range (-1,1)
        x: +1,-1 correspnds to +5mm,-5mm movement
        y: +1,-1 corresponds to +5mm,-5mm movement
        z: +1,-1 corresponds to apex, base 
        """
        
        # 1. Convert actions from output to positions
        # if z == 1 base else apex (eg 1=apex; 2 = base)
        if actions[-1] == -1:
            test_coords = self.apex_coords
        else:
            test_coords = self.base_coords 
        
        # Update x,y positions! 
        self.grid_pos[:,0:2] += actions[0:2]*5
        self.world.observe_mm[:,0:2] += actions[0:2]*5
        
        print(f"{self.grid_pos}, {self.world.observe_mm}")
        
        print('chicken')
        # 2. Update position, based on actions
        
        # 3. Compute new observaiton (sagittal, axial slices)
        
        # 4. Compute reward based on observations 
            # ie is lesion visible? 
            
        # 5. Return info, state, reward
        
                # Sample action x,y,z 
        
        # Get observation at x,y,z positions 
        # (sagital, axial) us
        # projection axial gland, target, sagittal gland, target
        
        # Update position; sample CCL from target lesion

        # Return rewards
        
        pass 
    
    
        
    def compute_reward(self):
        """
        Computes reward based on conditions met
        
        Notes:
        ----------
        a) Lesion observed : Whether a lesion can be seen in US view 
        b) Lesion targeted : Whether lesion is sampled effectively (based on depth)
        
        Additionally, to deal with sparse rewards, we can include shaped reward:
        a) Compute distance from current pos to target lesion 
        """
        pass 
    
    ############### HELPER FUNCTIONS   ###############
    
    def initialise_grid_coords(self):
        """
        Initialises apex and base needle sample coords for sampling!!1
        
        Returns:
        apex_mesh : 20 x 13 x 13 x 3 
        base_mesh : 20 x 13 x 13 x 3 
        
        Note:
        ---------
        20 needle samples; 13 x 13 grid points ; 3 x,y,z coords 
        """
        
        self.x_grid = torch.arange(-30,35,5)
        self.y_grid = torch.arange(-30,35,5)
        
        # Obtain target coords 
        self.prostate_coords = self.world.get_mask_coords_mm(self.world.gland)
        self.target_coords = self.world.get_mask_coords_mm(self.world.target)
        
        # target centroid
        self.target_centroid = torch.mean(self.target_coords, dim = 0)
        #target_z = self.target_coords[:,-1].min()
        
        # Grid centre is prostate_centroid
        # Returns in x,y,z coords
        self.prostate_centroid = torch.mean(self.prostate_coords, dim = 0)
        max_prostate = self.prostate_coords[:,:].max(dim=0)[0]
        min_prostate = self.prostate_coords[:,:].min(dim=0)[0]
        
        # z depths for needle sampling (apex and base )
        max_z = self.prostate_coords[:,-1].max() 
        mid_gland = self.prostate_coords[:,-1].mean()
        min_z = self.prostate_coords[:,-1].min()
        apex = (min_z + mid_gland)/2 # mid way between beginning of gland, and mid-gland
        base = (max_z + mid_gland)/2 # midway between base of gland base
        
        # Compute mesh grids for 13 x 13 x 2 grid 
        needle_length = 20 
        needle_samples = 20 
        z_apex =  torch.linspace(
                        apex - needle_length / 2,
                        apex + needle_length / 2,
                        needle_samples,
                    )
        z_base = torch.linspace(
                        base - needle_length / 2,
                        base + needle_length / 2,
                        needle_samples,
                    )
        
        # Mesh for obtaining coords for apex / base 
        # in z, x, y form!!! 
        apex_mesh = torch.stack(torch.meshgrid(z_apex, self.y_grid.float(), self.x_grid.float()), dim=3)[
                        ..., [2, 1, 0]
                    ]
        apex_mesh_centred = apex_mesh + self.prostate_centroid
        base_mesh = torch.stack(torch.meshgrid(z_base, self.y_grid.float(), self.x_grid.float()),dim=3)[
                        ..., [2, 1, 0]
                    ]  
        base_mesh_centred = base_mesh + self.prostate_centroid
        
        
    
        # os.makedirs("IMGS", exist_ok = True)
        # fig, axs = plt.subplots(1,2)
        # slice_num = int(64.0 + target_z)
        # gland_slice = self.world.target.squeeze()[slice_num,:,:].numpy()
        # us_slice = self.world.us.squeeze()[slice_num,:,:].numpy()
        # axs[0].imshow(us_slice)
        # axs[0].axis('off')
        # axs[1].imshow(gland_slice)
        # axs[1].axis('off')
        # plt.savefig(f"IMGS/target_us_{slice_num}.png")

        return apex_mesh_centred, base_mesh_centred
        
  
    def sample_ax_sag(self, img_vol, pos):
        """
        Samples axial / sagittal slice, given position of x,y,z in grid / depth
        """
        pass
    
    def compute_current_pos(self):
        pass 
    
    # def initialise_coords_mm(self, gland):
    #     # Initialise coords based on mm 
    #     pass
    
    #     # Compute COM of prostate ; centre (0,0,0) here
    #     self.grid_centre = torch.stack(
    #             [gland_coords_mm[b].mean(dim=0) for b in range(world.batch_size)],
    #             dim=0,
    #         )
    
    def get_reference_grid_mm(self):
        # reference_grid_*: (N, D, H, W, 3)
        # N.B. ij indexing for logical indexing, align_corners=True
        return torch.stack(
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
                        torch.linspace(
                            -self.vol_size_mm[n][0] / 2,
                            self.vol_size_mm[n][0] / 2,
                            self.vol_size[0],
                        ),
                        indexing="ij",
                    ),
                    dim=3,
                )
                for n in range(self.batch_size)
            ],
            dim=0,
        ).to(self.device)[
            ..., [2, 1, 0]
        ]  # ijk -> xyz

    def get_mask_coords_mm(self, mask):
        # mask: (b,1,z,y,x)
        # return a list of coordinates
        return [
            self.reference_grid_mm[b, mask.squeeze(dim=1)[b, ...], :]
            for b in range(self.batch_size)
        ]
            
    def initialise_state(self):
        
        # Start at the centre of the grid
        
        # Initialise grid coords based on sampled cords
        pass 
    
    def reset(self):
        """
        Resets all environment variables
        """  
        pass 
 
if __name__ == '__main__':
    
    DATA_FOLDER = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality'
    CSV_PATH = '/Users/ianijirahmae/Documents/PhD_project/Biopsy_RL/patient_data_multiple_lesions.csv'
    H5_PATH = '/Users/ianijirahmae/ImitationLearning/biopsy_dataset.h5'
    MODEL_TYPE = 'pix2pix'
    
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        MAP_LOCATION = 'gpu'
    else:
        DEVICE = torch.device("cpu")
        MAP_LOCATION = 'cpu'
        
        
    # Initialising datasets 
    MR_DATASET = MR_dataset(DATA_FOLDER, CSV_PATH, mode = 'train')
    MR_DL = DataLoader(MR_DATASET, batch_size = 1)
    h5_file = h5py.File(H5_PATH, 'r')
    voxdims_all = h5_file['voxdims_all'][()]
    # Initialising models 
    if MODEL_TYPE == 'pix2pix':
        #MODEL_PATH = '/Users/ianijirahmae/Documents/PhD_project/Experiments/mr2us_exp/220124/pix2pix/gen_model.pth'
        MODEL_PATH = '/Users/ianijirahmae/Documents/PhD_project/Experiments/mr2us_exp/new_pix2pix/Gen-520.pth'
        gen_model = Generator() 
    elif MODEL_TYPE == 'diffusion':
        raise NotImplementedError("TODO: not implemented yet")
    else:
        MODEL_PATH = '/Users/ianijirahmae/Documents/PhD_project/Experiments/mr2us_exp/220124/transformnet/best_val_model.pth'
        gen_model  = TransformNet()
    
    gen_model.load_state_dict(torch.load(MODEL_PATH, map_location = cpu))
    mr2us = MR2US(gen_model)
    
    
    ### Loading each dataset 

    for idx, voxdims in enumerate(voxdims_all):

        # Load each volume 
        gland = torch.tensor(
            h5_file["/gland_%04d" % idx][()], dtype=torch.bool, device=DEVICE
        )
        targets = torch.tensor(
            h5_file["/targets_%04d" % idx][()], dtype=torch.uint8, device=DEVICE
        )

        mr = torch.tensor(
            h5_file["/mr_%04d" % idx][()], device=DEVICE
        )

        # Obtain US: 
        mr_transposed = mr.permute(1, 2, 0)
        norm_mr = normalise_data(mr_transposed)
        us_norm = mr2us.convert_img(norm_mr)
        us_norm = us_norm.permute(0,1,4,2,3)
        us = mr2us.convert_img(mr_transposed) # in form 1 x 1 x h x w x d 
        us = us.permute(0,1,4,2,3) # convert back to 1 x 1 x d x h x w

        # Treat target as a batch size 
        num_t = targets.max()
        
        target_env = TargettingEnv(mr, 
                                   us, 
                                   targets, 
                                   gland)
        
        print('fuecoco')
        
        
        # tpb_envs = TPBEnv(
        #     mr = mr[None, None].expand(num_t, -1, -1, -1, -1), # expand to 4 x 1 x 96 x 200 x 200 
        #     us = us.expand(num_t, -1, -1, -1, -1),
        #     gland=gland[None, None].expand(num_t, -1, -1, -1, -1),
        #     target=torch.stack([targets == (i + 1) for i in range(num_t)]).unsqueeze(1),
        #     voxdims=[voxdims[::-1].tolist()] * num_t,
        # )