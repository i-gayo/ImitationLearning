import torch 
from Biopsy_env_single import * 
import numpy as np
from utils_il import * 
from stable_baselines3 import DDPG, PPO, SAC, DQN, TD3

def compute_norm_coverage(fired_pos, lesion_hit, lesion_obs):
    """
    A function that computes coverage as std(x)*std(y) * pi of all fired needle positions

    """
    
    # only compute coverage of needles that actually hit lesion 
    lesion_hit = np.array(lesion_hit)
    
    fired_x = fired_pos[lesion_hit == True,0]
    fired_y = fired_pos[lesion_hit == True,1]
    
    std_x = np.std(fired_x)
    std_y = np.std(fired_y)
    
    if (std_x == 0) and (std_y != 0):
        std_x = 1
    
    if (std_y == 0) and (std_x != 0):
        std_y = 1
    
    fired_area = (std_x * std_y * np.pi) # area of ellipse 
    
    # Compute max area
    lesion_obs =  np.max(lesion_obs[:,:,:,].numpy(), axis = 2) # maximum projeciton 
    lesion_area = np.sum(lesion_obs) # count number of non-zero pixels as apporximate to 2D area 

    norm_area = fired_area / lesion_area 
    
    return norm_area

def get_wacky_points(lesion_vol, strategy = 'edges'):
    
    """
    A function that obtains 4 grid grid points corresponding to a wacky strategy 
    
    Parameters:
    ----------
    lesion_vol(ndarray) : 200 x 200 x 96 binary mask of lesion volume 
    strategy (str) : indicates which type of points to obtain : edges or box 
    
    Returns:
    ----------
    grid_points : 4 points (4 x 2) corresponding to BR, UR, UL, BL
    
    """
    # lesion 2d projection in xy plane 
    lesion_proj = np.max(lesion_vol, axis = 2)
    all_y, all_x = np.where(lesion_proj)
    mean_x = np.mean(all_x)
    needle_grid = np.zeros_like(lesion_proj)
    
    if strategy == 'edges': 
        
        #tl
        min_x = np.min(all_x)
        corr_y = np.max(all_y[all_x == min_x])
        
        #bl
        max_y = np.max(all_y[all_x < mean_x])
        corr_x = np.min(all_x[all_y == max_y]) # left most 
        
        #br
        max_x = np.max(all_x)
        corr_ymax = np.max(all_y[all_x == max_x])
        
        #tr 
        min_y = np.min(all_y[all_x >= mean_x])
        corr_xmax = np.max(all_x[all_y == min_y])
    
        needle_grid[max_y ,corr_x] = 1 #bl
        needle_grid[corr_y, min_x] = 1 # tl 
        needle_grid[corr_ymax, max_x] = 1 # br 
        needle_grid[min_y, corr_xmax] = 1 # tr
        
        # obtain array of br, tr, tl, bl 
        coords = np.array([[corr_ymax, max_x], [min_y, corr_xmax], [corr_y, min_x], [max_y, corr_x]])
        
    else: # bounding box 
        
        lower_y = all_y[all_x <= mean_x]
        upper_y = all_y[all_x > mean_x]
        
        lower_x = np.min(all_x)
        ul_y = np.min(lower_y)
        bl_y = np.max(lower_y)

        upper_x = np.max(all_x)
        ur_y = np.min(upper_y)
        lr_y = np.max(upper_y)
        
        # plotting grid 
        needle_grid[ul_y ,lower_x] = 1 # tl
        needle_grid[bl_y, lower_x] = 1 # bl 
        needle_grid[ur_y, upper_x] = 1 # tr 
        needle_grid[lr_y, upper_x] = 1 # br
        
        # obtain array of br, tr, tl, bl 
        coords = np.array([[upper_x, lr_y], [upper_x, ur_y], [lower_x, ul_y], [lower_x, bl_y]])
        #coords = np.array([[lr_y, upper_x], [ur_y, upper_x], [ul_y, lower_x], [bl_y, lower_x]])
    
    return coords, needle_grid 

class ActionFinder_wacky():
    """
    Clas that finds relative actions; based on following assumptions:
    
    prostate_centroid is centre of grid (usually 100 x 100)
    
    """
    def __init__(self):
        super().__init__()
        # Define action maps to use to refine actions and take discrete actions between (10,10)
        self.action_maps = {3 : np.array([2,1]), 4 : np.array([2,2]), 5 : np.array([2,2,1]), 6 : np.array([2,2,2]), \
            7 : np.array([2,2,2,1]), 8 : np.array([2,2,2,2]), 9: np.array([2,2,2,2,1]), 10: np.array([2,2,2,2,2]), \
                11: np.array([2,2,2,2,2, 1]), 12: np.array([2,2,2,2,2,2]), 13: np.array([2,2,2,2,2,2,1])}
    
        # self.prostate_centroid = prostate_centroid
        # self.lesion_centroid = lesion_centroid
        # self.needle_depth = needle_depth
    
    def generate_grid(self, prostate_centroid):
        """
        Generates 2D grid of grid point coords on image coordinates
        
        Arguments:
        :prostate_centroid (ndarray) : centroid in x,y,z convention of prostate gland 
        
        Returns:
        :grid_coords (ndarray) : 2 x 169 grid coords x,y convention 
        """
        x_grid = (np.arange(-30,35,5))*2 + prostate_centroid[0]
        y_grid = (np.arange(-30,35,5))*2 + prostate_centroid[1]

        grid = np.zeros((200,200))
        for i in range(-60, 65, 10):
            for j in range(-60, 65, 10):
                grid[prostate_centroid[1]+j , prostate_centroid[0] +i] = 1

        grid_coords = np.array(np.where(grid == 1))  # given in y, x 
        
        # change to x,y convention instead of y,x 
        grid_coords[[0,1],:] = grid_coords[[1,0],:]
        
        return grid_coords 
    
    def get_wacky_points(self,lesion_vol, strategy = 'edges'):
    
        """
        A function that obtains 4 grid grid points corresponding to a wacky strategy 
        
        Parameters:
        ----------
        lesion_vol(ndarray) : 200 x 200 x 96 binary mask of lesion volume 
        strategy (str) : indicates which type of points to obtain : edges or box 
        
        Returns:
        ----------
        grid_points : 4 points (4 x 2) corresponding to BR, UR, UL, BL
        
        """
        # lesion 2d projection in xy plane 
        lesion_proj = np.max(lesion_vol, axis = 2)
        all_y, all_x = np.where(lesion_proj)
        mean_x = np.mean(all_x)
        needle_grid = np.zeros_like(lesion_proj)
        
        if strategy == 'edges': 
            
            #tl
            min_x = np.min(all_x)
            corr_y = np.max(all_y[all_x == min_x])
            
            #bl
            max_y = np.max(all_y[all_x < mean_x])
            corr_x = np.min(all_x[all_y == max_y]) # left most 
            
            #br
            max_x = np.max(all_x)
            corr_ymax = np.max(all_y[all_x == max_x])
            
            #tr 
            min_y = np.min(all_y[all_x >= mean_x])
            corr_xmax = np.max(all_x[all_y == min_y])
        
            needle_grid[max_y ,corr_x] = 1 #bl
            needle_grid[corr_y, min_x] = 1 # tl 
            needle_grid[corr_ymax, max_x] = 1 # br 
            needle_grid[min_y, corr_xmax] = 1 # tr
            
            # obtain array of br, tr, tl, bl 
            coords = np.array([[corr_ymax, max_x], [min_y, corr_xmax], [corr_y, min_x], [max_y, corr_x]])
            
        else: # bounding box 
            
            lower_y = all_y[all_x <= mean_x]
            upper_y = all_y[all_x > mean_x]
            
            lower_x = np.min(all_x)
            ul_y = np.min(lower_y)
            bl_y = np.max(lower_y)

            upper_x = np.max(all_x)
            ur_y = np.min(upper_y)
            lr_y = np.max(upper_y)
            
            # plotting grid 
            needle_grid[ul_y ,lower_x] = 1 # tl
            needle_grid[bl_y, lower_x] = 1 # bl 
            needle_grid[ur_y, upper_x] = 1 # tr 
            needle_grid[lr_y, upper_x] = 1 # br
            
            # obtain array of br, tr, tl, bl 
            coords = np.array([[upper_x, lr_y], [upper_x, ur_y], [lower_x, ul_y], [lower_x, bl_y]])
            #coords = np.array([[lr_y, upper_x], [ur_y, upper_x], [ul_y, lower_x], [bl_y, lower_x]])
        
        return coords, needle_grid 

    def get_grid_points(self, lesion_centroid, grid_coords, lesion_mask, num_points = 6):
        
        """
        Obtains the n closest grid points to the lesion centre 
        
        Arguments:
        : lesion_centroid (ndarray) : 1 x 3 lesion centroid x,y,z 
        : grid (ndarray) : 2 x 169 coords 
        : num_points (int) : num points to return 
        
        Returns:
        : closest_point (ndarray) : n x 2 closest points x,y convention
        """
        
        def get_closest_point(point):
            """
            Returns closest coordinate 
            """
            dif = grid_coords - point.reshape(2,1)
            dist = np.linalg.norm(dif, axis = 0)
            idx_order = np.argsort(dist)
            coords = grid_coords[:,idx_order[0]]
            
            return coords 
            
        # Get wacky points 
        wacky_points, needle_grid = self.get_wacky_points(lesion_mask, 'box')
        
        # Initialise empty array for each grid point 
        num_points, _ = np.shape(wacky_points)
        grid_points = np.zeros((num_points+1, 2))
        
        # assign first grid point as grid point closest to lesion centre 
        grid_points[0,:] = get_closest_point(lesion_centroid[0:-1].reshape(2,1))
        
        # Find closest grid point to each WACKY POINT 
        for i, point in enumerate(wacky_points):
            grid_points[i+1,:] = get_closest_point(point)
            print(f"grid point : {grid_points[i,:]}")
        
        closest_points = np.transpose(grid_points)
        
        return closest_points
    
    def compute_actions(self, grid_pos, start_pos):
        """
        Computes the action per time step by subtracting grid_pos by each time step grid pos
        ie delta_x, delta_y = grid_pos{t+1} - grid_pos{t}
        
        Parameters:
        :grid_pos (ndarray): n x 2 where n is number of timesteps
        :start_pos (ndarray): starting pos to compute from
        
        Returns:
        actions (ndarray) : raw actions based on img coords 
        """ 
        dif_points = np.zeros_like(grid_pos)
        dif_points[:,0] = start_pos[0:-1] # starting point
        dif_points[:, 1:] = grid_pos[:, 0:-1]
        actions = grid_pos - dif_points #dif_points - grid_pos 
        
        return actions 
    
    def refine_actions(self, raw_actions, depth):
        """
        Adds firing action delta_z
        Changes larger actions (greater than 10mm or 2 grid positions) to smaller discrete ones

        """

        # sacle acitons from -30,30
        actions = raw_actions / 10 # divide by 2 and 5 to get from -60,60 to 6,6
        num_needles = np.shape(actions)[1]

        refined_actions = [] 

        for idx in range(num_needles):

            act = actions[:,idx]
            paired_actions = [] 

            # Loop through each action x,y 
            for indv_act in act: 

                abs_indv_act = np.abs(indv_act)
                sign = np.sign(indv_act)
                
                # Loop through each x,y action 
                # If size of action is greater than 2 (ie greater than 10mm) -> split up actions to individual actions 
                if abs_indv_act > 2:
                    split_actions = sign * self.action_maps[abs_indv_act]
                else: 
                    split_actions = np.array([indv_act])
                
                paired_actions.append(split_actions)

            # Create array of split_actions : ie from [4, 3] -> [[2,2], [2,1]] to split up max movement 
            len_actions = [len(action) for action in paired_actions]
            new_actions = np.zeros([3, np.max(len_actions)])
            
            new_actions[0, 0:len_actions[0]] = paired_actions[0]
            new_actions[1, 0:len_actions[1]] = paired_actions[1]
            
            # Hit at the end of the split actions (ie when reached actual needle destination)
            new_actions[2, -1] = depth

            refined_actions.append(new_actions)
        
        # Concatenate actions
        refined_actions = np.concatenate(refined_actions, axis = 1)
        
        return refined_actions  

    def obtain_actions(self, prostate_centroid, lesion_centroid, needle_depth):
        
        grid_coords = self.generate_grid(prostate_centroid)
        grid_pos = self.get_grid_points(lesion_centroid, grid_coords)
        raw_actions = self.compute_actions(grid_pos, prostate_centroid)
        refined_actions = self.refine_actions(raw_actions, needle_depth)
        
        return raw_actions, refined_actions 
    
    def normalise_actions(self, actions):
        """
        Normalise acitons between -1,1 for training purposes 
        """
        
        nav_actions = actions[0:2,:]
        hit_actions = actions[-1,:]
        
        nav_actions = nav_actions / 2 # to get between -1,1
        hit_actions = hit_actions - 1 # between -1,1 
        
        norm_actions = np.zeros_like(actions)
        norm_actions[0:2, :] = nav_actions
        norm_actions[-1, :] = hit_actions
        
        return norm_actions 
            
    def __call__(self, prostate_centroid, lesion_centroid, needle_depth, lesion_mask):
        """
        Returns raw, refined and normalised actions upon calling class 
        """
        
        grid_coords = self.generate_grid(prostate_centroid)
        grid_pos = self.get_grid_points(lesion_centroid, grid_coords, lesion_mask)
        raw_actions = self.compute_actions(grid_pos, prostate_centroid)
        refined_actions = self.refine_actions(raw_actions, needle_depth)
        norm_actions = self.normalise_actions(refined_actions)
        
        return raw_actions, refined_actions, norm_actions 

class ActionFinder_gs():
    """
    Clas that finds relative actions; based on following assumptions:
    
    prostate_centroid is centre of grid (usually 100 x 100)
    
    """
    def __init__(self):
        super().__init__()
        # Define action maps to use to refine actions and take discrete actions between (10,10)
        self.action_maps = {3 : np.array([2,1]), 4 : np.array([2,2]), 5 : np.array([2,2,1]), 6 : np.array([2,2,2]), \
            7 : np.array([2,2,2,1]), 8 : np.array([2,2,2,2]), 9: np.array([2,2,2,2,1]), 10: np.array([2,2,2,2,2]), \
                11: np.array([2,2,2,2,2, 1]), 12: np.array([2,2,2,2,2,2]), 13: np.array([2,2,2,2,2,2,1])}
    
        # self.prostate_centroid = prostate_centroid
        # self.lesion_centroid = lesion_centroid
        # self.needle_depth = needle_depth
    
    def generate_grid(self, prostate_centroid):
        """
        Generates 2D grid of grid point coords on image coordinates
        
        Arguments:
        :prostate_centroid (ndarray) : centroid in x,y,z convention of prostate gland 
        
        Returns:
        :grid_coords (ndarray) : 2 x 169 grid coords x,y convention 
        """
        x_grid = (np.arange(-30,35,5))*2 + prostate_centroid[0]
        y_grid = (np.arange(-30,35,5))*2 + prostate_centroid[1]

        grid = np.zeros((200,200))
        for i in range(-60, 65, 10):
            for j in range(-60, 65, 10):
                grid[prostate_centroid[1]+j , prostate_centroid[0] +i] = 1

        grid_coords = np.array(np.where(grid == 1))  # given in y, x 
        
        # change to x,y convention instead of y,x 
        grid_coords[[0,1],:] = grid_coords[[1,0],:]
        
        return grid_coords 
    
    def get_grid_points(self, lesion_centroid, grid_coords, num_points = 6):
        """
        Obtains the n closest grid points to the lesion centre 
        
        Arguments:
        : lesion_centroid (ndarray) : 1 x 3 lesion centroid x,y,z 
        : grid (ndarray) : 2 x 169 coords 
        : num_points (int) : num points to return 
        
        Returns:
        : closest_point (ndarray) : n x 2 closest points x,y convention
        
        """
        
        dif_to_centroid = grid_coords - lesion_centroid[0:-1].reshape(2,1)
        dist_to_centroid = np.linalg.norm(dif_to_centroid, axis = 0)
        
        # sort in order of closeness to lesion centroid
        idx_order = np.argsort(dist_to_centroid)
        ordered_points = grid_coords[:,idx_order]
        closest_points = ordered_points[:,0:num_points]
        
        print(f'Closest points : {closest_points}')
        return closest_points
        
    def compute_actions(self, grid_pos, start_pos):
        """
        Computes the action per time step by subtracting grid_pos by each time step grid pos
        ie delta_x, delta_y = grid_pos{t+1} - grid_pos{t}
        
        Parameters:
        :grid_pos (ndarray): n x 2 where n is number of timesteps
        :start_pos (ndarray): starting pos to compute from
        
        Returns:
        actions (ndarray) : raw actions based on img coords 
        """ 
        dif_points = np.zeros_like(grid_pos)
        dif_points[:,0] = start_pos[0:-1] # starting point
        dif_points[:, 1:] = grid_pos[:, 0:-1]
        actions = grid_pos - dif_points #dif_points - grid_pos 
        
        return actions 
    
    def refine_actions(self, raw_actions, depth):
        """
        Adds firing action delta_z
        Changes larger actions (greater than 10mm or 2 grid positions) to smaller discrete ones

        """

        # sacle acitons from -30,30
        actions = raw_actions / 10 # divide by 2 and 5 to get from -60,60 to 6,6
        num_needles = np.shape(actions)[1]

        refined_actions = [] 

        for idx in range(num_needles):

            act = actions[:,idx]
            paired_actions = [] 

            # Loop through each action x,y 
            for indv_act in act: 

                abs_indv_act = np.abs(indv_act)
                sign = np.sign(indv_act)
                
                # Loop through each x,y action 
                # If size of action is greater than 2 (ie greater than 10mm) -> split up actions to individual actions 
                if abs_indv_act > 2:
                    split_actions = sign * self.action_maps[abs_indv_act]
                else: 
                    split_actions = np.array([indv_act])
                
                paired_actions.append(split_actions)

            # Create array of split_actions : ie from [4, 3] -> [[2,2], [2,1]] to split up max movement 
            len_actions = [len(action) for action in paired_actions]
            new_actions = np.zeros([3, np.max(len_actions)])
            
            new_actions[0, 0:len_actions[0]] = paired_actions[0]
            new_actions[1, 0:len_actions[1]] = paired_actions[1]
            
            # Hit at the end of the split actions (ie when reached actual needle destination)
            new_actions[2, -1] = depth

            refined_actions.append(new_actions)
        
        # Concatenate actions
        refined_actions = np.concatenate(refined_actions, axis = 1)
        
        return refined_actions  

    def obtain_actions(self, prostate_centroid, lesion_centroid, needle_depth):
        
        grid_coords = self.generate_grid(prostate_centroid)
        grid_pos = self.get_grid_points(lesion_centroid, grid_coords)
        raw_actions = self.compute_actions(grid_pos, prostate_centroid)
        refined_actions = self.refine_actions(raw_actions, needle_depth)
        
        return raw_actions, refined_actions 
    
    def normalise_actions(self, actions):
        """
        Normalise acitons between -1,1 for training purposes 
        """
        
        nav_actions = actions[0:2,:]
        hit_actions = actions[-1,:]
        
        nav_actions = nav_actions / 2 # to get between -1,1
        hit_actions = hit_actions - 1 # between -1,1 
        
        norm_actions = np.zeros_like(actions)
        norm_actions[0:2, :] = nav_actions
        norm_actions[-1, :] = hit_actions
        
        return norm_actions 
            
    def __call__(self, prostate_centroid, lesion_centroid, needle_depth):
        """
        Returns raw, refined and normalised actions upon calling class 
        """
        
        grid_coords = self.generate_grid(prostate_centroid)
        grid_pos = self.get_grid_points(lesion_centroid, grid_coords)
        raw_actions = self.compute_actions(grid_pos, prostate_centroid)
        refined_actions = self.refine_actions(raw_actions, needle_depth)
        norm_actions = self.normalise_actions(refined_actions)
        
        return raw_actions, refined_actions, norm_actions 
    
class SimBiopsyEnv():
    """
    A simulated biopsy env, using functions from RL Biopsy env
    
    # Class methods: 
    find_new_needle_pos()
    
    obtain_new_obs()
    
    create_needle_vol()
    
    compute_HR()
    
    compute_CCL()
    
    """
    
    def __init__(self, device):
        self.device = device
        
    def find_new_needle_pos(self, actions, grid_pos):
        """
        Obtains new grid positions, given current actions 
        
        Arguments:
        :actions (tensor): 3x1 tensor of chosen actions
        :prev_pos (tensor): 2x1 previous position 
        
        
        Returns:
        :new_pos (tensor) : new position on grid, along with suitable depth 
        """
        
        # Convert action z into depths 
        action_z = actions[:,2].to(self.device)
        depths = torch.zeros_like(action_z).to(self.device)
        non_fired = [action_z <=-0.33]
        apex = (action_z > -0.33) * (action_z <= 0.33)
        base = action_z > 0.33 
        
        depths[non_fired] = 0 
        depths[apex] = 1
        depths[base] = 2
        
        # Convert x,y actions 
        print(f"grid_pos {grid_pos} action{actions}")
        max_step_size = 10 # changed max step size to 10 
        prev_pos = grid_pos - torch.tensor([50,50]).to(self.device) # subtract by origin of 50,50 

        action_x = actions[:,0]
        action_y = actions[:,1]

        x_movement = round_to_05(action_x * max_step_size)
        y_movement = round_to_05(action_y * max_step_size)
        print(f"x_movement {x_movement} y {y_movement}")
        updated_x = (prev_pos[:,0] + x_movement)
        updated_y = (prev_pos[:,1] + y_movement)
        
        #Dealing with boundary positions 
        x_lower = updated_x < -30
        x_higher = updated_x > 30
        y_lower = updated_y < -30
        y_higher = updated_y > 30

        if torch.any(x_lower).item():
            #Change updated_x to maximum     
            #updated_x =  -30
            updated_x[x_lower] = -30

        if torch.any(x_higher).item():
            updated_x[x_higher] =  30

        if torch.any(y_lower).item(): 
            updated_y[y_lower] = -30

        if torch.any(y_higher).item(): 
            updated_y[y_higher] = 30

        x_grid = updated_x.int()
        y_grid = updated_y.int()

        new_pos = torch.stack([(x_grid), (y_grid), depths], dim = 1)

        return new_pos
            
    def obtain_new_obs(self, current_pos, max_depth):
      """
      Obtains new needle pos given the new grid position 

      Parameters:
      ----------
      current_pos : 3 x 1 array (x,y,z) where z is (0,1,2) where 0 i snon-fired, 1 is apex and 2 is base 
      
      Returns:
      ----------
      obs : batch_size x 100 x 100 x 25 observations 
      
      """
      batch_size = current_pos.size()[0]
      needle_obs = torch.zeros([batch_size, 100, 100, 24])
      
      if batch_size == 1:
          grid_pos = current_pos[0,:]
          needle_obs[0,:,:,:] = self.create_needle_vol(grid_pos, max_depth)
      else:
        for i in range(batch_size):
            grid_pos = current_pos[i,:]
            needle_obs[i, :,:,:] = self.create_needle_vol(grid_pos, max_depth[i])

    
      # Find new obs empty 
      
      if len(np.unique(needle_obs)) == 1:
          print(f'Empty needle vol action {current_pos} max depth {max_depth}')
      
      return needle_obs
   
    def create_needle_vol(self, current_pos, max_depth):

      """
      A function that creates needle volume 100 x 100 x 24 

      Parameters:
      -----------
      current_pos : current position on the grid. 1 x 3 array delta_x, delta_y, delta_z ; assumes in the range of (-30,30) ie actions are multipled by 5 already 

      Returns:
      -----------
      neeedle_vol : 100 x 100 x 24 needle vol for needle trajectory 

      """
      needle_vol = torch.zeros([100,100,24])

      x_idx = current_pos[0]#*5 +50
      y_idx = current_pos[1]#*5 +50

      #Converts range from (-30,30) to image grid array
      #x_idx = (x_idx) + 50
      #y_idx = (y_idx) + 50

      x_grid_pos = int(x_idx)
      y_grid_pos = int(y_idx)

      depth_map = {0 : 1, 1 : int(0.5*max_depth), 2 : max_depth}
      depth = depth_map[int(current_pos[2])]
      #print(f"Depth : {depth} current pos : {current_pos[2]}")

      needle_vol[y_grid_pos-1:y_grid_pos+ 2, x_grid_pos-1:x_grid_pos+2, 0:depth] = 1

      if len(np.unique(needle_vol)) == 1:
          print('chicken')
      
      return needle_vol 

    def compute_HR(self, lesion_obs, needle_obs, pred_actions):
        """
        Compute both HR and CCL using observations 

        Arguments:
        :lesion_obs (tensor) : batch_size x 100 x 100 x 24 
        :needle_obs (tensor) : batch_Size x 100 x 100 x 24
        :pred_actions : batch_size x3 

        """

        # clone actions and obs 
        obs_img = lesion_obs

        # current needle pos 
        batch_size = needle_obs.size()[0]
        lesion_obs = lesion_obs.to(self.device)
        needle_obs = needle_obs.to(self.device)
        # find which idx are fired
        fired_idx = torch.where(pred_actions[:,-1] >= -0.33)[0]
        NUM_FIRED = len(fired_idx)

        # from matplotlib import pyplot as plt 
        # check_idx = 5
        # #plt.figure()
        # fig, axs = plt.subplots(2)
        # axs[0].imshow(np.max(lesion_obs[fired_idx[check_idx],:,:,:].numpy(), axis = 2))
        # axs[1].imshow(np.max(needle_obs[fired_idx[check_idx],:,:,:].numpy(), axis = 2))
        
        # no fired needles -> no hit arrays, no ccl arrays 
        if NUM_FIRED == 0:
        
            print(f'No fire needles : HR = 0, CCL = 0')
            ccl_array = torch.tensor([0], dtype = torch.float32)
            hit_array = torch.tensor([0], dtype = torch.float32) 
            
            return hit_array, ccl_array
        
        else:
            
            # compute intsersections between new needle obs and lesion 
            intersect = lesion_obs[fired_idx, :, :, :] * needle_obs[fired_idx,  :, :, :]
            all_lesions = lesion_obs[fired_idx, :,:,:]

            # Indices of all the hit patients 
            all_hit_idx = (torch.unique(torch.where(intersect)[0]))

            # Compute HR 
            HR = (len(all_hit_idx) / NUM_FIRED) * 100

            # hit ar
            hit_array = torch.zeros(len(fired_idx))
            hit_array[all_hit_idx] = 1
            ccl_array = [] 
            
            # Compute CCL:
            for hit_idx in range(NUM_FIRED):
                if hit_array[hit_idx] == 0: 
                    ccl = 0 
                    norm_ccl = 0 
                else: 
                    z_depths = torch.where(intersect[hit_idx, :,:,:])[-1]
                    min_z = torch.min(z_depths)
                    max_z = torch.max(z_depths)
                    
                    # lesion depths 
                    z_lesion = torch.where(all_lesions[hit_idx, :, :, :])[-1]
                    min_z_lesion = torch.min(z_lesion)
                    max_z_lesion = torch.max(z_lesion)
                
                    # added 1 : to account for each voxel being 0.25 in resolution; if only intersect in one voxel (8,8) then still ccl = 4mm  
                    ccl = (max_z+1 - min_z )*4
                    ccl_max = (max_z_lesion+1 - min_z_lesion )*4
                    
                    norm_ccl = ccl / ccl_max 

                ccl_array.append(norm_ccl)

                #print(f'{max_z} , {min_z}, {ccl}')

            CCL = torch.mean(torch.FloatTensor(ccl_array))
            ccl_std = torch.std(torch.FloatTensor(ccl_array))

            print(f"Hit rate : {HR} CCL {CCL} +/- {ccl_std} for num fired : {NUM_FIRED}")

            return torch.tensor(hit_array, dtype = torch.float32), torch.tensor(ccl_array, dtype = torch.float32)

def compute_grid_pos(obs):
    """
    Comptues curernt grid position, based on needle obs 
    To be used for comptuing HR / CCL after!!! 
    Args:
        obs (_type_): _description_
    """
    
    needle_mask = obs[:,-2,:,:,:] # use previous needle obs 
    batch_size = needle_mask.size()[0]
    
    current_grid_pos = torch.zeros([batch_size, 2])
    for i in range(batch_size):
        y,x,_ = torch.where(needle_mask[i, :,:])
        current_grid_pos[i, :] = torch.tensor([torch.mean(x.to(torch.float)), torch.mean(y.to(torch.float))])
    
    return current_grid_pos 
    
##### Functions to debug and test out gold standard functions 

# Initialising environment  

if __name__ == '__main__':
    
    #### THINGS TO CHANGE: 
    # Modify this to what you need / want 
    STRATEGY = 'WACKY'
    
    # Change to path on your own device 
    DATASET_PATH = '/Users/ianijirahmae/Documents/DATASETS/Data_by_modality'
    CSV_PATH = '/Users/ianijirahmae/Documents/PhD_project/MRes_project/Reinforcement Learning/patient_data_multiple_lesions.csv'
    
    
    ###### training parameters ###### 
    MAX_NUM_STEPS = 20
    MODE = 'train'
    LOG_DIR = 'debug'
    os.makedirs(LOG_DIR, exist_ok=True) 
    
    if STRATEGY == 'GS':
        LABEL_PATH = 'NEW_GS.h5'
    else:
        LABEL_PATH = 'NEW_WACKY.h5'
        
    # Generate h5py file 
    hf = h5py.File(LABEL_PATH, 'a')    

    PS_dataset = Image_dataloader(DATASET_PATH, CSV_PATH, use_all = True, mode  = MODE)
    Data_sampler = DataSampler(PS_dataset)

    # For environment, sample all training data
    Biopsy_env = TemplateGuidedBiopsy_single(Data_sampler, reward_fn = 'penalty', terminating_condition = 'more_than_5', \
    start_centre = True, train_mode = MODE, env_num = '1', max_num_steps = MAX_NUM_STEPS, results_dir= LOG_DIR, deform = False)
    SimEnv = SimBiopsyEnv(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    NUM_EPISODES = len(PS_dataset)
    print(f'Length of {MODE} : {NUM_EPISODES}')

    reward_per_episode = np.zeros(NUM_EPISODES)
    all_episode_len = np.zeros(NUM_EPISODES)
    #lesions_hit = np.zeros(NUM_EPISODES)
    hit_threshold = np.zeros(NUM_EPISODES)
    hit_rate = np.zeros(NUM_EPISODES)
    ccl_corr_vals = np.zeros(NUM_EPISODES)
    efficiency = np.zeros(NUM_EPISODES)
    plots = []
    all_ccl = []
    all_sizes = [] 
    all_norm_ccl = [] 
    all_coverage = [] 
    all_area = [] 

    for ep_num in range(NUM_EPISODES):
        
        print(f'\n Episode num {ep_num}')
        
        if ep_num == 0:
            obs = Biopsy_env.get_initial_obs()
        
        else:
            print(f'New episode, env reset')
            obs = Biopsy_env.reset()
                    
        # Obtain new patient 
        patient_name = Biopsy_env.get_patient_name()
        print(f'Patient name : {patient_name}')
        lesion_centroid, lesion = Biopsy_env.get_lesion_centroid()
        prostate_centroid = Biopsy_env.get_prostate_centroid()
        needle_depth = Biopsy_env.get_needle_depth()
        single_lesion_mask = Biopsy_env.get_single_lesion_mask()
        
        # Obtain ground truth acitons BASED ON WACKY or GS actions!
        if STRATEGY == 'GS':
            Actions = ActionFinder_gs()
            raw, refine, norm = Actions(prostate_centroid, lesion_centroid, needle_depth) #, single_lesion_mask)
        else:
            Actions = ActionFinder_wacky()
            raw, refine, norm = Actions(prostate_centroid, lesion_centroid, needle_depth, single_lesion_mask) #for wacky 
        
        NUM_STEPS = np.shape(norm)[-1]
        
        # Generate group name for h5 AND SAVE OBS AND ACTIONS 
        if patient_name in hf:
            group_folder = hf[patient_name]
            print(f'patient name : {patient_name}')
            already_exists = True
        else:
            already_exists = False
            group_folder = hf.create_group(patient_name)
        
        # Obtain actions from GS strategy 
        all_obs = torch.zeros([NUM_STEPS, 5, 100, 100, 24]) # num timesteps x 5 x 100 x 100 x 24
        all_actions = torch.transpose(torch.tensor(norm), 0,1)
        
        # add initial obs 
        all_obs[0,:,:,:,:] = obs 
        
        # METRICS 
        episode_reward = 0
        fired_pos = [] 
        lesion_hit = [] 
        
        for step in range(NUM_STEPS):
            
            print(f'\n Step num {step}')   
            action = norm[:,step]
            #action[1] = action[1]#*-1 # multiply by -1
            # print(f'action : {refine[:-1,step]}')
            # print(f'norm aciton : {action[:-1]}')
            
            obs, reward, done_new_patient, info = Biopsy_env.step(action)
            
            # action : action at t = 0, obs at t=1
            hit = (reward == 700)
            
            #grid_pos = grid_pos.to(device)å
            prostate = obs[1,:,:,:]
            max_depth = np.max(np.where(prostate == 1)[2])
            
            # Compute hr, ccl from predicted actions 
            actions = torch.tensor(action)
            grid_pos = compute_grid_pos(obs.unsqueeze(0)) # Compute needle position 
            print(f"0.Grid_pos : {grid_pos}")
            print(f"Actions {actions}")
            expected_pos = grid_pos + actions[0:-1]*10
            
            print(f"1.Grid_pos : {grid_pos} expected pos {expected_pos}")
            # To get from new pos to expected pos : new_pos*5 + 50
            new_pos = SimEnv.find_new_needle_pos(actions.unsqueeze(0), grid_pos)
            print(f"1.new pos {new_pos}")
            new_pos[0,0] = grid_pos[0][0] + actions[0]*10
            new_pos[0,1] = grid_pos[0][1] + actions[1]*10
            print(f"2. Grid pos {grid_pos}")
            print(f"2.new pos {new_pos}")
            needle_obs = SimEnv.obtain_new_obs(new_pos, max_depth)
            #print(f"3. Grid pos {grid_pos}")
            print(f"Biopsy env : HIT {hit}")
            hr, ccl = SimEnv.compute_HR(obs[0,:,:,:].unsqueeze(0), needle_obs, actions.unsqueeze(0))

            # DEBUGGING PLOTS 
            # if (hit == True) and (hr == 0):
            #     print(f"Current pos : {grid_pos} action {actions} new pos {new_pos}")
            #     fig, axs = plt.subplots(1,3)
            #     axs[0].imshow(np.max((obs[0,:,:,:] + obs[-2,:,:,:]).numpy(), axis =2))
            #     axs[1].imshow(np.max((obs[0,:,:,:] + obs[-1,:,:,:]).numpy(), axis =2))
            #     axs[2].imshow(np.max((needle_obs*10 + obs[0,:,:,:]).squeeze().numpy(), axis = 2))                
            #     print('chicken')
            #     plt.close()
                
            # Save obs to all obs : ie next obs corresponds to next aciton 
            if step != NUM_STEPS-1:
                all_obs[step+1, :,:,:,:] = obs 
            
            needle = obs[4,:,:,:]
            lesion = obs[0,:,:,:]

            # Obtain metrics 
            episode_reward += reward
            norm_ccl = info['norm_ccl']
            all_norm_ccl.append(norm_ccl)
            current_pos = info['current_pos']
                
            if action[-1] >= -0.33: # ie fired position
                fired_pos.append(current_pos[:-1])
                lesion_hit.append(info['needle_hit'])

        # Write data into h5py file 
        if already_exists:
            pass
        else:
            group_folder.create_dataset('all_actions', data = all_actions)
            group_folder.create_dataset('all_obs',data = all_obs)
            group_folder.create_dataset('tumour_centroid', data =lesion_centroid)
            group_folder.create_dataset('prostate_centroid', data =prostate_centroid)
            group_folder.create_dataset('lesion_mask',data = single_lesion_mask)
            
        #### METRIC :  compute coverage 
        if len(fired_pos) != 0: 
            all_fired_pos = np.stack(fired_pos)
            coverage = compute_norm_coverage(all_fired_pos, lesion_hit, obs[0,:,:,:])
        else: 
            # no needles fired 
            coverage = 0
                
        all_coverage.append(coverage)
        
        # Save episode reward 
        reward_per_episode[ep_num] = episode_reward
        hit_threshold[ep_num] = info['hit_threshold_reached']
        hit_rate[ep_num] = info['hit_rate']
        ccl_corr_vals[ep_num] = info['ccl_corr_online']
        efficiency[ep_num] = info['efficiency']
        all_ccl.append(info['all_ccl'])
        all_sizes.append(info['all_lesion_size'])

        print(f"Episode reward : {episode_reward}")
        print(f"Hit rate : {info['hit_rate']}")
        print(f"Num needles per lesion : {info['num_needles_per_lesion']}")
        print(f"Correlation coeff {info['ccl_corr_online']}")
        print(f"Efficiency {info['efficiency']}")
            
    # COMPUTE METRICS ACROSS ALL EPISODES 
    average_episode_reward = np.nanmean(reward_per_episode)
    std_episode_reward = np.nanstd(reward_per_episode)
    average_episode_len = np.nanmean(all_episode_len)

    avg_hit_rate = np.nanmean(hit_rate) * 100 
    std_hit_rate = np.nanstd(hit_rate)
    avg_ccl_corr = np.nanmean(ccl_corr_vals)
    std_corr = np.nanstd(ccl_corr_vals)
    avg_efficiency = np.nanmean(efficiency) # just hit rate, but / 100
    avg_hit_threshold = np.nanmean(hit_threshold) * 100 # average threshold reached ie 4 needles per lesion achieved

    # Print coverage metrics: 
    avg_norm_ccl = np.nanmean(all_norm_ccl) 
    std_norm_ccl = np.nanstd(all_norm_ccl)
    all_norm_coverage = np.array(all_coverage)
    avg_norm_coverage = np.nanmean(all_norm_coverage[~np.isinf(all_norm_coverage)])
    std_norm_coverage = np.nanstd(all_norm_coverage[~np.isinf(all_norm_coverage)])

    print(f"Average episode reward {average_episode_reward} +/- {std_episode_reward}")
    print(f"Average episode length {average_episode_len}")
    #print(f"Average percentage of lesions hit {average_percentage_hit}")
    print(f"Average correlation coefficient {avg_ccl_corr}")
    print(f"Average Efficiency {avg_efficiency} +- {std_hit_rate}")
    print(f"Aveage norm ccl : {avg_norm_ccl} +- {std_norm_ccl}")
    print(f"Average norm coverage : {avg_norm_coverage} +- {std_norm_coverage}")

    hf.close() 

print('chickne')


