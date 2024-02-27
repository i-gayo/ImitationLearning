### Computes a single biopsy env based on previous env
from utils.mr2us_utils import * 
from utils.data_utils import * 
from environment.biopsy_env import * 
import time 
import h5py 
import copy
import gym 
import numpy as np
from gym import spaces 

def map_range(value, old_min=0, old_max=1, new_min=1, new_max=2):
    # Linear mapping formula
    return int(((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min)


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
    
class TargettingEnv(gym.Env):
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
        
    Notes:
            # Get observation at x,y,z positions 
        # (sagital, axial) us
        # projection axial gland, target, sagittal gland, target
        # Update position; sample CCL from target lesion
        # Return rewards
        
    """      
    def __init__(self, data_sampler, max_steps = 20, deform = False, device = torch.device('cpu')):
        
        # Initialising action and observation space
        self.action_space = spaces.MultiDiscrete(np.array([3,3,2]))
        self.observation_space = spaces.Box(low=0, high=1.0,
                                    shape=(6, 120,128), dtype=np.float32)

        # Sample first patient 
        self.data_sampler = data_sampler 
        self.device = device
        
        # Initialise counting positions for termination 
        self.step_counter = 0 
        self.max_steps = 20 
        self.slices_count = 0 # how many observed slices with target are found! 
        self.min_slices = 3 # how many "slices" to observe before terminating episode 
        self.deform = deform # whether to deform or not 
        
        # Initialise idx map 
        self.idx_map = {}
        for idx, val in zip(np.arange(0,14), np.arange(-30,35,5)):
            self.idx_map[str(val)] = idx
        
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
        z: +1,-1 corresponds to base, apex  
        """

        # 1. Convert actions from output to positions
        # if z == 1 base else apex (eg 1=apex; 2 = base)
        if actions[-1] == 0:
            test_coords = self.apex_coords
            z_depth = self.apex
            sample = 'apex'
        else:
            test_coords = self.base_coords 
            z_depth = self.base 
            sample = 'base'
            
        # 2. Update position, based on actions
        # TODO: add depth for image slicing!! 
        # Save grid position 

        # self.grid_pos[:,0:2] += actions[0:2]*5
        # self.grid_pos[:,-1] = map_range(actions[-1])
        # self.prev_pos_mm = copy.deepcopy(self.world.observe_mm) # Save previous position 
        # self.world.observe_mm[:,0:2] += actions[0:2]*5
        # self.world.observe_mm[:,-1] = z_depth
        # print(f"{self.grid_pos}, {self.world.observe_mm}")
        # Convert actions from (0,1) -> -1,1 

        self.update_pos(actions, z_depth)
        self.step_counter += 1 # Increase step counter 
        
        #print('chicken')
        # 3. Compute new observaiton (sagittal, axial slices)
        # Deform observatinos first 
        self.transition.update(self.world, self.action)
        obs = self.observation.update(self.world, sample, self.step_counter).to(self.device)

        # TODO: 3. Compute CCL metrics, given x,y and depth 
        ccl = self.compute_ccl(obs, test_coords)
        #print(f"Idx : {self.step_counter-1} : CCL sampled : {ccl}")
        # 4. Compute reward based on observations 
            # ie is lesion visible? 
            # is lesion sampled (for needle reward or metrics only) -> CCL
            # include shaped reward : closer to lesion target 
        
        reward, contains_lesion = self.compute_reward(obs) 
        
        print(f"Step : {self.step_counter-1}  CCL sampled : {ccl} Reward : {reward}")
        
        if contains_lesion:
            self.slices_count += 1
            
        # 6. Check if terminated 

        exceeds_step = (self.step_counter >= self.max_steps)
        lesion_found = (self.slices_count >= self.min_slices) # if found at least min_slices containing lesion
        terminated = exceeds_step or lesion_found 
        
        # 5. Return info, state, reward
        truncated = None 
        info = {'num_steps' : self.step_counter, 
                'slices_count' : self.slices_count,
                'prev_pos' : self.prev_pos_mm,
                'current_pos' : self.world.observe_mm,
                'grid_pos' : self.grid_pos, 
                'ccl' : ccl.item()}
        
        return obs.squeeze().cpu(), reward, terminated, info

    def reset(self):
        """
        Resets all environment variables
        """  
        # Initialise counters to 0 
        self.step_counter = 0 
        self.slices_count = 0 
        
        # Initialise new patient 
        initial_obs = self.initialise_world()
        
        return initial_obs.cpu()
    
    ############### REWARD FUNCTIONS   ###############
    
    def get_target(self):
        """
        Returns target and computes centroid 
        """
        
        img_coords = torch.nonzero(self.world.target.squeeze(), as_tuple=True)
        mm_coords = self.target_centroid
        
        return self.world.target, img_coords, mm_coords 
            
    def update_pos(self, raw_actions, z_depth):
        
        # Normalise actions in ranges required 
        actions = np.zeros_like(raw_actions)
        actions[0] = map_range(raw_actions[0],old_min=0, old_max=1, new_min=-1, new_max=1)
        actions[1] = map_range(raw_actions[1],old_min=0, old_max=1, new_min=-1, new_max=1)
        actions[2] = map_range(raw_actions[2], old_min = 0, old_max =1 , new_min = 1, new_max = 2)
        
        # Save previous position 
        self.prev_pos_mm = copy.deepcopy(self.world.observe_mm)
        
        x_check = self.grid_pos[:,0] + actions[0]*5
        y_check = self.grid_pos[:,1] + actions[1]*5
        
        # Do not update x if more than 30 
        
        if x_check > 30:
            print(f"Outside grid boundaries x=+30")
            x_check = 30 
        elif x_check < -30:
            print(f"Outside grid boundaries x=-30")
            x_check = -30 
        else:
            # Update if not within boundaries
            self.world.observe_mm[:,0] += actions[0]*5
            
        if y_check > 30:
            print(f"Outside grid boundaries y=+30")
            y_check = 30 
        elif y_check < -30:
            print(f"Outside grid boundaries y=-30")
            y_check = -30 
        else:
            # Update if not within boundaries
            self.world.observe_mm[:,1] += actions[1]*5
        
        # Set depths for observe_mm, grid pos respectively 
        self.grid_pos[:,0:2] = torch.tensor([x_check,y_check])
        self.grid_pos[:,-1] = actions[-1] # changed from map_ragnge as this is done above code 
        self.world.observe_mm[:,-1] = z_depth
        
        print(f"New positions {self.grid_pos}, {self.world.observe_mm}")
        
            
    def compute_ccl(self, obs, needle_coords):
        """
        Computes CCL
        """  
        
        # from grid_pos 
        grid_pos = self.grid_pos 
        x_pos = str(grid_pos[0][0].item())
        y_pos = str(grid_pos[0][1].item())
        
        # [0,20]is good but not depth!
        # needle_coords_norm = batch_size x 20 x 1 x 1 x 3 where 20 is depths; 3 is x,y,z
        needle_sampled_coords = needle_coords[:,self.idx_map[x_pos], self.idx_map[y_pos],:].unsqueeze(1).unsqueeze(1).unsqueeze(0)
        #needle_coords_norm : needle_coords_norm ([4, 20, 1, 1, 3]) choose grid position we are currently at! 
        
        print(f"Needle sampled coords : {needle_sampled_coords.squeeze().item()}")
        
        needle_sampled = sampler(
            self.world.target.type(torch.float32), needle_sampled_coords)
        
        # Combined length for each batch of images 
        self.ccl_sampled = needle_sampled.squeeze().sum(dim=0)
        
        return self.ccl_sampled 
    
    def compute_reward(self, obs, inc_shaped = False, inc_needle = False):
        """
        Computes reward based on conditions met
        
        Notes:
        ----------
        a) Lesion observed : Whether a lesion can be seen in US view 
        b) Lesion targeted : Whether lesion is sampled effectively (based on depth)
        
        Additionally, to deal with sparse rewards, we can include shaped reward:
        a) Compute distance from current pos to target lesion 
        """
        REWARD = 0 

        # Computes reward based on whether lesion is observed in sagittal slice 
        sagittal_target = obs[:,-1,:,:]
        # axial_target = obs[:,-2,:,:]
        
        # # Debugging plotting 
        # from matplotlib import pyplot as plt 
        # fig, axs = plt.subplots(1,2)
        # axs[0].imshow(sagittal_target.squeeze().numpy())
        # axs[1].imshow(axial_target.squeeze().numpy())
        # plt.savefig("IMGS/TARGET-NEW.png")
        
        # Check whether sagittal contains lesion 
        contains_lesion = torch.any(sagittal_target != 0)
        
        if contains_lesion:
            REWARD += 10
        else:
            REWARD -= 0.1 # small penalty for not finding lesion 

        if inc_needle:
            raise NotImplementedError("Not yet implemented needle reward")

        if inc_shaped:
            raise NotImplementedError("Not yet implemneted needle error")
        
        return REWARD, contains_lesion
    
    ############### HELPER FUNCTIONS   ###############
        
    def initialise_world(self):
        
        # Obtains world coordinates (in mm) and image coordinates
        mr, us, gland, target, target_type, voxdims = self.data_sampler.sample_data()
        # Move to device
        mr, us, gland, target = mr.to(self.device), us.to(self.device), gland.to(self.device), target.to(self.device)
        
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
        
        # Initialise which samples have been sampled! 
        self.sample_x = torch.full((13,), False, dtype=torch.bool)
        self.sample_y = torch.full((13,), False, dtype=torch.bool)
        self.sample_d = torch.full((2,), False, dtype = torch.bool)
        
        ##### Get initial observations : mid-gland, centre of grid 
        # coomment out action update as no actions yet 
        #self.action.update(self.world, self.observation)
            
        # Deform prostate and gland randomly (using parameters)
        if self.deform:
            self.transition.update(self.world, self.action)
        
        # Obtain initial obseravtions
        # consists of gland axial + sagital; target axial
        initial_obs = self.observation.update(self.world)
        
        #us = torch.stack(us)
        #initial_obs = [us,gland,target]§
        # construct initial obs : stack us, gland and target together 
        
        # initialise sample_x -30,30 -> 0- 13
        self.idx_map = {}
        for idx, val in zip(np.arange(0,14), np.arange(-30,35,5)):
            self.idx_map[str(val)] = idx
        
        # Testing something:
        #self.action.update(self.world, self.observation)
        
        # plt.figure()
        # plt.imshow(np.max(target.squeeze()[:,:,:].numpy(), axis = 2))
        # plt.savefig("IMGS/LESION_MASK.png")
        return initial_obs
         
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
        
        # Save apex and base coords
        self.apex = apex
        self.base = base
        
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
                    ].to(self.world.target.device)
        apex_mesh_centred = apex_mesh + self.prostate_centroid
        base_mesh = torch.stack(torch.meshgrid(z_base, self.y_grid.float(), self.x_grid.float()),dim=3)[
                        ..., [2, 1, 0]
                    ]  .to(self.world.target.device)
        base_mesh_centred = base_mesh + self.prostate_centroid
        
        # needle_coords_norm : normalised 
        self.unit_dims = self.world.unit_dims
        self.apex_mesh_centred_norm = apex_mesh_centred/self.unit_dims
        self.base_mesh_centred_norm = base_mesh_centred/self.unit_dims
    
        #')
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