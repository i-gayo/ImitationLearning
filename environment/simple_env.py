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
    
class TargettingEnv():
    
    """
    A simple biopsy env that follows simple MDP:
    
    Observation:
        6 images : U_ax, U_sag, T_ax, T_sag, P_ax, P_sag
    
    Actions : 
        3 actions : 
            x : (-5,+5)
            y : (-5,+5)
            z : (1,2) # base / apex sampling
    """
    
    def __init__(self, **kwargs):
        self.world = LabelledImageWorld_with_US(**kwargs)
        # self.mr = mr 
        # self.us = us 
        # self.target = target
        # self.prostate = prostate 
        
        self.coords, self.grid = self.initialise_coords_mm(self.prostate)
        
    def step(self, actions):
        """
        Updates new observation
        Computes reward 
        """
        # 1. Convert actions from output to positions
        
        # 2. Update position, based on actions
        
        # 3. Compute new observaiton (sagittal, axial slices)
        
        # 4. Compute reward based on observations 
            # ie is lesion visible? 
            
        # 5. Return info, state, reward
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
        
    ############### HELPER FUNCTIONS   ###############
    
    def sample_ax_sag(self, img_vol, pos):
        """
        Samples axial / sagittal slice, given position of x,y,z in grid / depth
        """
        pass
    
    def compute_current_pos(self):
        pass 
    
    def initialise_coords_mm(self, gland):
        # Initialise coords based on mm 
        pass
    
        # Compute COM of prostate ; centre (0,0,0) here
        self.grid_centre = torch.stack(
                [gland_coords_mm[b].mean(dim=0) for b in range(world.batch_size)],
                dim=0,
            )
    
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
                                   glands)
        
        
        
        
        # tpb_envs = TPBEnv(
        #     mr = mr[None, None].expand(num_t, -1, -1, -1, -1), # expand to 4 x 1 x 96 x 200 x 200 
        #     us = us.expand(num_t, -1, -1, -1, -1),
        #     gland=gland[None, None].expand(num_t, -1, -1, -1, -1),
        #     target=torch.stack([targets == (i + 1) for i in range(num_t)]).unsqueeze(1),
        #     voxdims=[voxdims[::-1].tolist()] * num_t,
        # )