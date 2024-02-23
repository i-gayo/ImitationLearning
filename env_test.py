### Computes a single biopsy env based on previous env
from utils.mr2us_utils import * 
from utils.data_utils import * 
from environment.biopsy_env import * 
import time 
import h5py 
import nibabel as nib

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
    
    def __init__(self, **kwargs):
        
        # Obtains world coordinates (in mm) and image coordinates
        self.world = LabelledImageWorld(**kwargs) # initialises starting coordinaets 
        
        # Obtains current position in z,y,x
        self.current_pos = self.world.observe_mm

        # Initialises grid sampling positions 
        
            
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
        pass 
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

class MR_US_dataset_alllabels(Dataset):
    
    """
    Dataset that acquires the following: 
    MR
    MR_label
    US
    US_label 
    """
    
    def __init__(self, dir_name, mode = 'train', downsample = False, alligned = False, get_2d = False, target = True):
        self.dir_name = dir_name 
        self.mode = mode 
        
        # obtain list of names for us and mri labels 
        self.us_names = os.listdir(os.path.join(dir_name, mode, 'us_images'))
        self.us_label_names = os.listdir(os.path.join(dir_name, mode, 'us_labels'))
        self.mri_names = os.listdir(os.path.join(dir_name, mode, 'mr_images'))
        self.mri_label_names = os.listdir(os.path.join(dir_name, mode, 'mr_labels'))
        self.num_data = len(self.us_names)
        self.downsample = downsample
        self.alligned = alligned
        self.get_2d = get_2d # whether to get whole volume or 2d slices only
        self.target = target # Whether using targets (all of them) or some only 

        # Load items 
        self.us_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'us_images', self.us_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        self.us_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode,'us_labels', self.us_label_names[i])).get_fdata()) for i in range(self.num_data)]
        self.mri_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_images', self.mri_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        self.mri_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_labels', self.mri_label_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        
    def __len__(self):
        return self.num_data
        
    def __getitem__(self, idx):
        
        #upsample_us = self.resample(self.us_data[idx])
        #upsample_us_labels = self.resample(self.us_labels[idx], label = True)
        
        if self.alligned:
            # no need to transpose if alligned alerady
            t_us = self.us_data[idx]
            t_us_labels = self.us_labels[idx]
        else: # Change view of us image to match mr first 
            t_us = torch.transpose(self.us_data[idx], 2,0)
            t_us_labels = torch.transpose(self.us_labels[idx], 2,0)
            
        # Upsample to MR images 
        upsample_us = self.resample(t_us)
        upsample_us_labels = self.resample(t_us_labels, label = True)
        
        # Add dimesion for "channel"
        mr_data = self.mri_data[idx].unsqueeze(0)
        mr_label = self.mri_labels[idx].unsqueeze(0)
        #mr_label = mr_label[:,:,:,:,0]       # use only prostate label
        
        if len(mr_label.size()) > 4:
            if not(self.target): # choose targets if not already target
                mr_label = mr_label[:,:,:,:,(0,3,4,5)]       # use only prostate label
            # lesion : 4 ; calcifications 5, 6 (which might be blank but its okay)
        
        # Squeeze dimensions 
        us = upsample_us.squeeze().unsqueeze(0)
        us_label = upsample_us_labels.squeeze().unsqueeze(0)
        
        # normalise data 
        mr = self.normalise_data(mr_data)
        us = self.normalise_data(us)
        mr_label = self.normalise_data(mr_label)
        us_label = self.normalise_data(us_label)
        
        # if resample is true 
        if self.downsample: 
            upsample_method = torch.nn.Upsample(size = (64,64,64))
            mr = upsample_method(mr.unsqueeze(0)).squeeze(0)
            us = upsample_method(us.unsqueeze(0)).squeeze(0)
            mr_label = upsample_method(mr_label.unsqueeze(0)).squeeze(0)
            us_label = upsample_method(us_label.unsqueeze(0)).squeeze(0)
        
        if self.get_2d:
            # Returns only slice ie 1 x width x height only (axial direction obtains)
            num_slices = mr.size()[-1]
            # TODO ; option to obtain inner slices only
            slice_idx = np.random.choice(np.arange(0,num_slices-1))
            mr = mr[:,:,:,slice_idx]     
            mr_label = mr_label[:,:,:,slice_idx]
            us = us[:,:,:,slice_idx]
            us_label = us_label[:,:,:,slice_idx]
               
        return mr, us, mr_label, us_label
    
    def resample(self, img, dims = (120,128,128), label = False):
        upsample_method = torch.nn.Upsample(size = dims)
        if label: 
            
            if len(img.size()) == 4:
                if not (self.target):
                    img_label = img[:,:,:,(0,3,4,5)]
                else:
                    img_label = img[:,:,:,:]
            else:
                img_label = img 
                
            # # Choose only prostate gland label 
            # if len(img.size()) > 3:
            #     img_label = img[:,:,:,0]
            # else:
            #     img_label = img 
            
            if len(img.size()) == 4:
                img_to_upsample = img_label.unsqueeze(0)
            else:
                img_to_upsample = img_label.unsqueeze(0).unsqueeze(0)

            # needs in dimensions bs x channels x width x height x depth so turn into channel!!! 
            channel_img = img_to_upsample.permute(0, 4, 1,2,3)
            upsampled_img = upsample_method(channel_img)
            reordered_img = upsampled_img.permute(0, 2,3,4,1)
            return reordered_img
        else:
            upsampled_img = upsample_method(img.unsqueeze(0).unsqueeze(0))
        
        return upsampled_img
    
    def normalise_data(self, img):
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
     
        
class BiopsyDataset:
    """
    Dataset for sampling biopsy images 
    """
    def __init__(self, dir_name, mode, give_fake = False):
        self.dir_name = dir_name 
        self.mode = mode 
        
        # obtain list of names for us and mri labels 
            
        self.us_names = os.listdir(os.path.join(dir_name, mode, 'us_images'))
        self.us_label_names = os.listdir(os.path.join(dir_name, mode, 'us_labels'))
        self.mri_names = os.listdir(os.path.join(dir_name, mode, 'mr_images'))
        self.mri_label_names = os.listdir(os.path.join(dir_name, mode, 'mr_labels'))
        self.num_data = len(self.us_names)
        # Load folder path 
        
        # Load items 
        if give_fake: 
            print(f"Using fake images")
            folder_us = 'fake_us_images'
        else:
            print(f"Using real images")
            folder_us = 'us_images'
            
        self.us_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, folder_us, self.us_names[i])).get_fdata().squeeze()) for i in range(self.num_data)]
        self.us_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode,'us_labels', self.us_label_names[i])).get_fdata()) for i in range(self.num_data)]
        self.mri_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_images', self.mri_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        self.mri_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_labels', self.mri_label_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        
        # Load voxdims of data 
        test_img = (nib.load(os.path.join(dir_name, mode, folder_us, self.us_names[0])))
        self.voxdims = test_img.header.get_zooms()[1:]
        
        print('chicken')
        
    def __getitem__(self, idx):
        
        #upsample_us = self.resample(self.us_data[idx])
        #upsample_us_labels = self.resample(self.us_labels[idx], label = True)

        # no need to transpose if alligned alerady
        us = self.us_data[idx]
        us_label = self.us_labels[idx]

        # Add dimesion for "channel"
        mr_data = self.mri_data[idx].unsqueeze(0)
        mr_label = self.mri_labels[idx].unsqueeze(0)
        #mr_label = mr_label[:,:,:,:,0]       # use only prostate label
        
        # normalise data 
        mr = self.normalise_data(mr_data)
        us = self.normalise_data(us)
        mr_label = self.normalise_data(mr_label)
        us_label = self.normalise_data(us_label)

        gland = us_label[:,:,:,0]
        
        # Choose a target with non-zero vals        
        # Checks which targets are non-empty 
        t_idx = torch.unique(torch.where((mr_label[:,:,:,:,1:]) == 1)[-1])
        
        if len(t_idx) == 0:
            target = us_label[:,:,:,1].squeeze()
            target_type = 'None'
        else:
            # Randomly sample target / ROI
            if 0 in t_idx: # ie lesion available, use this 
                target = us_label[:,:,:,t_idx[0]].squeeze()
                target_type = 'Lesion'
            # if no lesion available, use calcification / 
            else:
                roi_idx = np.random.choice(t_idx)
                target = us_label[:,:,:,roi_idx].squeeze()
                target_type = 'Other'
            
            # TODO: change dimensions from height width depth to depth widht height for yipeng's code!!!!
            mr = self.change_order(mr)
            us = self.change_order(us.unsqueeze(0))
            gland = self.change_order(gland.unsqueeze(0))
            target = self.change_order(target.unsqueeze(0))
            
        return mr, us.unsqueeze(0), gland.unsqueeze(0), target.unsqueeze(0), target_type, self.voxdims
    
    def __len__(self):
        return self.num_data
    
    def normalise_data(self, img):
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
    
    def change_order(self, tensor_data):
        """
        Changes order of tensor from width height depth to edpth width height 
        """
        
        return tensor_data.permute(0,3,1,2)
    
        
        
        
        
if __name__ == '__main__':
    
    IMG_FOLDER = '/raid/candi/Iani/mr2us/ALL_DATA/RL_pix2pix'
    ds = BiopsyDataset(IMG_FOLDER, mode = 'test', give_fake = True)
    dl = DataLoader(ds, batch_size = 1)
    
    nonzero = 0 
    for idx,(mr, us, gland, target, target_type, voxdims) in enumerate(dl):
        
        target_type = target_type[0]
        print(f"{target.size()}")
        if target_type == 'None':
            pass 
        else:
            nonzero+=1
        print(f'idx {idx} : target : {target_type}')
        
    print('chicken')
        
        
    #img_path = os.path.join(IMG_DATA, 'case000011.nii.gz')
    
    # # Load the NIfTI image file
    # nifti_img = nib.load(img_path)  # Replace 'your_image_file.nii.gz' with your actual file path

    # # Access the header of the image
    # header = nifti_img.header

    # # Extract voxel dimensions from the header
    # voxel_dimensions = header.get_zooms()
    # print(f'fuecoco : {voxel_dimensions}')
    
    #DATA_FOLDER = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality'
    #H5_PATH = './data/biopsy_dataset.h5'
    #MODEL_TYPE = 'pix2pix'
    
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        MAP_LOCATION = 'gpu'
    else:
        DEVICE = torch.device("cpu")
        MAP_LOCATION = 'cpu'
        
    # Initialising datasets 
    h5_file = h5py.File(H5_PATH, 'r')
    voxdims_all = h5_file['voxdims_all'][()]

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

        print('chicken')

        # Obtain US: 
        # mr_transposed = mr.permute(1, 2, 0)
        # norm_mr = normalise_data(mr_transposed)
        # us_norm = mr2us.convert_img(norm_mr)
        # us_norm = us_norm.permute(0,1,4,2,3)
        # us = mr2us.convert_img(mr_transposed) # in form 1 x 1 x h x w x d 
        # us = us.permute(0,1,4,2,3) # convert back to 1 x 1 x d x h x w

        # Treat target as a batch size 
        num_t = targets.max()
        
        target_env = TargettingEnv(mr = mr[None, None].expand(num_t, -1, -1, -1, -1), # expand to 4 x 1 x 96 x 200 x 200 
            us = us.expand(num_t, -1, -1, -1, -1),
            gland=gland[None, None].expand(num_t, -1, -1, -1, -1),
            target=torch.stack([targets == (i + 1) for i in range(num_t)]).unsqueeze(1),
            voxdims=[voxdims[::-1].tolist()] * num_t,
        )  
        
        print('fuecoco')
        