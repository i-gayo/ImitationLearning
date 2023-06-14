import torch
from torch import nn
from torch.nn import functional as F

# Import helper functions from vit and efficient network libraries 
from utils_efficient import * 
from utils_vit import * 

class EfficientNet3D(nn.Module):

    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet3D.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None, in_channels=3):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv3d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock3D(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock3D(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layers 
        self._avg_pooling = nn.AdaptiveAvgPool3d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)

        # output for action, value net parameters 
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

        #self.action_net = nn.Sequential(nn.Linear(features_dim, num_actions), nn.Tanh())
        #self.value_net = nn.Sequential(nn.Linear(features_dim, 1))


    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        # TODO : if idx >= 11: break ; provide features!!! :) 
        
        for idx, block in enumerate(self._blocks):
            #print(f"Block : {idx} \n")
            if idx >= 11:
                break
            
            else:
                drop_connect_rate = self._global_params.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self._blocks)
                x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers

        # pad input 
        p1d = (0, 38) # padded to ensure consistency with layers 
        x = F.pad(inputs, p1d, "constant", 0)
        x = self.extract_features(x)

        if self._global_params.include_top:
            # Pooling and final linear layer
            x = self._avg_pooling(x)
            x = x.view(bs, -1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None, in_channels=3):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params, in_channels)

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """ 
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

class ImitationConv(nn.Module):
    """
    # Input : 1 x 100 x 100 x 75 
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        Correspods to the number of unit for the last layer.
    """

    def __init__(self, num_input_channels = 3, features_dim: int = 512, multiple_frames = False, num_multiple_frames = 3, num_actions = 3):
        
        super(ImitationConv, self).__init__()
        # Assumes CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        num_input_channels = 5
        self.cnn_layers = nn.Sequential(

            # First layer like resnet, stride = 2
            nn.Conv3d(num_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            # Apply pooling layer in between 
            nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            # Apply pooling layer in between 
            nn.MaxPool3d(kernel_size = 3, stride = 2, padding = 1),
            nn.Conv3d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        #Flatten layers 
        self.flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            all_layers = nn.Sequential(self.cnn_layers, self.flatten)
            
            #observation_space_shuffled = np.transpose(observation_space.sample(), [2, 1, 0])
            #n_flatten = all_layers(torch.as_tensor(observation_space_shuffled[None]).float()).shape[1]
            #processed_obs_space = self._pre_process_image(torch.zeros))).float()
            processed_obs_space = torch.zeros([1, 5, 100, 100, 24])
            n_flatten = all_layers(processed_obs_space).shape[1]  
        
        # Feature Extractor 
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        
        # Output layers for converting from features_dim -> num_actions. Output = Mean of each action 
        #self.final_linear = nn.Sequential(nn.Linear(features_dim, num_actions ), nn.Tanh())
        
        #self.action_net = nn.Sequential(nn.Linear(features_dim, num_actions), nn.Tanh())
        #self.value_net = nn.Sequential(nn.Linear(features_dim, 1))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:


        # FEATURE EXTRACTOR 
        observations = observations.float() 
        output = self.cnn_layers(observations)
        output = self.flatten(output)
        output = self.linear(output)
        
        #output = self.final_linear(output)

        # ACTION AND VALUE EXTRACTOR 
        #action_output = self.action_net(output)
        #value_output = self.value_net(output)

        return output

class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
           nn.LayerNorm(dim),
           nn.Linear(dim, num_classes)
        )
          
    def forward(self, video):
        """
        Input required is in format : [batch_size, num_channels, z_depth, x, y] but our input is in format [batch_size, num_channels, x, y, z_depth] so need to transpose!
        """
        video_transpose = video.permute(0, 1, 4, 2,3)
        x = self.to_patch_embedding(video_transpose)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x

class ActorCritic_network(nn.Module):
    """
    Actor-critic network, that uses same feature extractor for actor and critic heads 
    action_net outputs num_batch_size x 3 where 3 is x,y,z actions
    value_net outputs predicted value of network 
    
    Actor_critic_network takes in as input a feature extractor, which is pre-defined
    Either simple conv, efficient net or a vit network. 
    
    """
    def __init__(self, FeaturesExtractor, in_channels=5, num_actions = 3, features_dim = 512):
        super(ActorCritic_network, self).__init__()
        #self.FeatureExtractor = EfficientNet3D.from_name("efficientnet-b1",  override_params={'num_classes': features_dim}, in_channels=in_channels)
        
        self.features_extractor = FeaturesExtractor
        self.policy_net = nn.Sequential(nn.Linear(features_dim, num_actions), nn.Tanh())
        self.value_net = nn.Sequential(nn.Linear(features_dim, 1))

    def forward(self, observations: torch.Tensor):
                # ACTION AND VALUE EXTRACTOR 
        observations = observations.float()
        features = self.features_extractor(observations)
        #action_output = self.action_net(features)
        #value_output = self.value_net(features)

        return self.forward_actor(features), self.forward_critic(features), features
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)
    
if __name__ == '__main__':
    
    FEATURES_DIM = 512
    INPUT_CHANNELS = 5 

    # Defining feature extractors for use in network architecture 
    FeaturesExtractor_simple = ImitationConv()
    FeaturesExtractor_efficientnet = EfficientNet3D.from_name("efficientnet-b0",  override_params={'num_classes': FEATURES_DIM}, in_channels=INPUT_CHANNELS)
    Features_ViT = ViT(
    image_size = 100,          # image size
    frames = 24,               # number of frames
    image_patch_size = 20,     # image patch size # changed to 20 from 16
    frame_patch_size = 2,      # frame patch size
    num_classes = FEATURES_DIM, #output channel final dimensions 
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1, 
    channels = 5)
    
    # Define actor-critic networks
    model_simple = ActorCritic_network(FeaturesExtractor_simple)
    model_efficient = ActorCritic_network(FeaturesExtractor_efficientnet)
    model_vit = ActorCritic_network(Features_ViT)
    
    # Example predictions to test models work 
    #preds = Features_ViT(video)
    img = torch.randn(4, 5, 100, 100, 24)
    preds_vit, value_vit, _ = model_vit(img)
    preds_efficient, value_efficient, _ = model_efficient(img)
    preds_conv, value_conv, _ = model_simple(img)
    
    print('chicken ')