import torch 
from torch import nn as nn 
import numpy as np 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym 

class UNet(torch.nn.Module):

    def __init__(self, input_channel = 1, output_channel = 1, num_features = 32, num_layers = 3):

        super(UNet, self).__init__()

        feature_size = num_features * np.array([2**i for i in range(num_layers +1)]) 
        self.num_layers = num_layers
        len(feature_size)

        # Define UNet network 
        self.encoders = []
        self.decoders = [] 
        self.maxpools = [nn.MaxPool3d(kernel_size = 2, stride = 2) for i in range(num_layers)]
        self.upconvs = [] 

        self.encoders.append(self.build_conv_block(input_channel = 1, num_features = feature_size[0]))
        
        for i in range(0, num_layers-1):
            #print(f' Num features {feature_size[i]} and output : {feature_size[i+1]}')
            self.encoders.append(self.build_conv_block(feature_size[i],feature_size[i+1]))
            self.decoders.append(self.build_conv_block(feature_size[i+1],feature_size[i+1]))
            self.upconvs.append(nn.ConvTranspose3d(feature_size[i], feature_size[i+1], kernel_size = 2, stride = 2))

        # Bottle neck is last layer to be used to link the two together 
        self.bottle_neck = self.build_conv_block(feature_size[-2],feature_size[-1])
        print(f'Bottle neck : {feature_size[-2]}, {feature_size[-1]}')

        #for i in range(num_layers, 0, -1):
        #    print(f' Num features {feature_size[i]} and output : {feature_size[i-1]}')
        #    self.decoders.append(self.build_conv_block(feature_size[i],feature_size[i-1]))
        #    self.upconvs.append(nn.ConvTranspose3d(feature_size[i], feature_size[i-1], kernel_size = 2, stride = 2))

        self.final_conv = nn.Conv3d(num_features, output_channel, kernel_size = 1)

    def forward(self, x):

        enc_layers = []
        dec_layers = [] 

        for i in range(self.num_layers):
            if i == 0:            
                x = self.encoders[0](x)
            else:
                x = self.maxpools[i](self.encoders[i](x))
            enc_layers.append(x)

        # bottleneck
        x = self.bottle_neck(self.maxpools[-1](x))

        # decoder layers
        for i in range(self.num_layers, -1, 1):
                print(i)
                x = self.upconvs[i](x)
                x = torch.cat(x, enc_layers[i], dim = 1)
                x = self.decoders[i](x)
            
        x = torch.sigmoid(self.final_conv(x))

        return x 

    def build_conv_block(self,input_channel, num_features):

        conv_block = nn.Sequential(
            nn.Conv3d(input_channel, out_channels = num_features, kernel_size = 3, bias = False), 
            nn.BatchNorm3d(num_features),
            nn.ReLU(inplace = True),
            nn.Conv3d(num_features, num_features, kernel_size = 3, bias = False),
            nn.BatchNorm3d(num_features), 
            nn.ReLU(inplace = True)
        )

        return conv_block 

class UNet_v2(nn.Module):

    def __init__(self, input_channel = 1, output_channel = 1, num_features = 32, num_layers = 4):
        super(UNet_v2, self).__init__()

        self.num_features = num_features 

        # Identify each layers in the UNet
        self.encoder_1 = UNet_v2._build_conv_block(input_channel, num_features)
        self.maxpool_1 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder_2 = UNet_v2._build_conv_block(num_features, num_features*2)
        self.maxpool_2 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder_3 = UNet_v2._build_conv_block(num_features*2, num_features*4)
        self.maxpool_3 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder_4 = UNet_v2._build_conv_block(num_features*4, num_features*8)
        self.maxpool_4 = nn.MaxPool3d(kernel_size = 2, stride = 2)

        self.bottle_neck = UNet_v2._build_conv_block(num_features *8, num_features * 16)

        self.upconv_4 = nn.ConvTranspose3d(num_features * 16, num_features * 8, kernel_size = 2, stride = 2)
        self.decoder_4 = UNet_v2._build_conv_block((num_features*8)*2, num_features *8)
        self.upconv_3 = nn.ConvTranspose3d(num_features * 8, num_features * 4, kernel_size = 2, stride = 2)
        self.decoder_3 = UNet_v2._build_conv_block((num_features*4)*2, num_features *4)
        self.upconv_2 = nn.ConvTranspose3d(num_features * 4, num_features * 2, kernel_size = 2, stride = 2)
        self.decoder_2 = UNet_v2._build_conv_block((num_features*2)*2, num_features *2)
        self.upconv_1 = nn.ConvTranspose3d(num_features*2 , num_features, kernel_size = 2, stride = 2)
        self.decoder_1 = UNet_v2._build_conv_block(num_features*2, num_features)

        self.final_conv = nn.Conv3d(num_features, output_channel, kernel_size = 1)

        # to change dimensions of final output to 13 x 13 x 2 
        self.final_maxpool = nn.MaxPool3d(kernel_size = (9, 9, 24))

    def forward(self, x):
        
        enc1 = self.encoder_1(x)
        enc2 = self.encoder_2(self.maxpool_1(enc1))
        enc3 = self.encoder_3(self.maxpool_2(enc2))
        enc4 = self.encoder_4(self.maxpool_3(enc3))

        bottleneck = self.bottle_neck(self.maxpool_4(enc4))

        dec4 = self.upconv_4(bottleneck)

        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder_4(dec4)

        dec3 = self.upconv_3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder_3(dec3)

        dec2 = self.upconv_2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder_2(dec2)

        dec1 = self.upconv_1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder_1(dec1)

        # Final maxpool 3d applied to depth dimension only 
        layer_2d = self.final_maxpool(dec1)

        # Cut to first 13 x 13 layers
        return self.final_conv(layer_2d)
        #return torch.sigmoid(self.final_conv(layer_2d))

    @staticmethod
    def _build_conv_block(input_channel, num_features):
        
        conv_block = nn.Sequential(
            nn.Conv3d(in_channels = input_channel, out_channels = num_features, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm3d(num_features = num_features),
            nn.ReLU(inplace= True),
            nn.Conv3d(in_channels = num_features, out_channels = num_features, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm3d(num_features = num_features),
            nn.ReLU(inplace=True))

        return conv_block 

class ImitationNetwork(nn.Module):
    """
    # Input : 1 x 100 x 100 x 75 
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        Correspods to the number of unit for the last layer.
    """

    def __init__(self, num_input_channels = 3, features_dim: int = 512, multiple_frames = False, num_multiple_frames = 3, num_actions = 3):
        
        super(ImitationNetwork, self).__init__()
        # Assumes CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        num_input_channels = num_input_channels#rows x cols x channels 
        num_multiple_frames = num_multiple_frames
        self.num_multiple_frames = num_multiple_frames
        self.cnn_layers = nn.Sequential(

            # First layer like resnet, stride = 2
            nn.Conv3d(num_multiple_frames, 32, kernel_size=3, stride=2, padding=1),
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
            processed_obs_space = self._pre_process_image(torch.zeros([1, 100, 100, 75])).float()
            n_flatten = all_layers(processed_obs_space).shape[1]  

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        # Output layers for converting from features_dim -> num_actions. Output = Mean of each action 
        self.final_linear = nn.Sequential(nn.Linear(features_dim, num_actions ), nn.Tanh())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if len(np.shape(observations)) == 4:
            observations = self._pre_process_image(observations.squeeze(0))
        elif len(np.shape(observations)) == 7:
            observations = self._pre_process_image(observations.squeeze(0))
        else:
            observations = self._pre_process_image(observations.unsqueeze(0))
        observations = observations.float() 
        
        output = self.cnn_layers(observations)
        output = self.flatten(output)
        output = self.linear(output)
        output = self.final_linear(output)
    
        return output

    def _pre_process_image(self, images):
        """ 
        A function that changes dimensions of input obs from 1 x 100 x 100 x 75 -> 1 x 3 x 100 x 100 x 25!!

        C
        """ 
        with torch.no_grad():
            image = images.clone().detach().to(torch.uint8)#.unsqueeze(0)
            if len(image.size()) == 3:
                image = image.unsqueeze(axis = 0)
            split_channel_image = torch.cat([torch.cat([image[j,:,:,i*25:(i*25)+25].unsqueeze(0) for i in range(3)]).unsqueeze(0) for j in range(image.size()[0])])#.clone().detach().to(torch.uint8)
            #split_channel_image = torch.cat([torch.cat(torch.tensor_split(image[i,:,:,:].unsqueeze(0), self.num_multiple_frames, dim=3)).unsqueeze(0) for i in range(image.size()[0])])
            #processed_image = image.permute(0, 3,2,1)
            #processed_image = torch.squeeze(split_channel_image) #, dim= 0)
            
        # Turn image from channel x row x column -> channel x row x column x depth for pre-processing with 3D layers 

        return split_channel_image

if __name__ == '__main__':

    Network = UNet_v2(1, 1, 32, 3)
    output = Network(torch.zeros((1, 1, 96, 256,256)))
    print(output.size())
    
    # When computing own loss function, only use middle 200x200 grid, NOT entire image 
    loss_fn = nn.BCELoss()

    output_zeros = torch.zeros((200,200))
    loss_val = loss_fn(output.squeeze().squeeze()[28:-28,28:-28], output_zeros)

    print('Chicken')