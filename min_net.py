import torch
import torch.nn as nn


'''
This was created for the purpose of debugging. Has some simplifications.
Namely, no residual connections from the encoder to the decoder. There
are possibly others that I forgot but this shouldn't be used for training.
'''
class ResNetBlock(nn.Module):
  def __init__(self, channels):
    super(ResNetBlock, self).__init__()
    # 1. using 32 groups which is the default from GN paper
    #
    # 2. nn.ReLU() creates an nn.Module which you can add e.g. 
    # to an nn.Sequential model. nn.functional.relu on the 
    # other side is just the functional API call to the relu 
    # function, so that you can add it e.g. in your forward 
    # method yourself.
    #
    # Generally speaking it might depend on your coding style 
    # if you prefer modules for the activations or the functional calls. 
    # Personally I prefer the module approach if the activation has an 
    # internal state, e.g. PReLU.
    #
    self.feats = nn.Sequential(nn.GroupNorm(32, channels),
        nn.ReLU(inplace=True),
        #nn.Dropout3d(),
        nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(32, channels),
        nn.ReLU(inplace=True),
        #nn.Dropout3d(),
        nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1))

  def forward(self, x):
    residual = x
    out = self.feats(residual)
    return out


class DownSampling(nn.Module):
    # downsample by 2; simultaneously increase feature size by 2
    def __init__(self, in_channels):
        super(DownSampling, self).__init__()
        self.conv3x3x3 = nn.Conv3d(in_channels, 2*in_channels, 
            kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv3x3x3(x)

class CompressFeatures(nn.Module):
    # Reduce the number of features by a factor of 2.
    # Assumes channels_in is power of 2.
    def __init__(self, channels_in, channels_out):
        super(CompressFeatures, self).__init__()
        self.conv1x1x1 = nn.Conv3d(channels_in, channels_out,
                kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv1x1x1(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=3, stride=1, padding=1),
            ResNetBlock(32),
            DownSampling(32),
            ResNetBlock(64),
            ResNetBlock(64),
            DownSampling(64),
            ResNetBlock(128), 
            ResNetBlock(128), 
            DownSampling(128),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256), 
            ResNetBlock(256)
            )
        
        self.decoder = Decoder()

    def forward(self, x):
        x=self.encoder(x)
        x=self.decoder(x) 
        return x

class Decoder(nn.Module):
    def __init__(self, output_channels=3):
        super(Decoder, self).__init__()
        self.sig = nn.Sigmoid()
        self.cf1 = CompressFeatures(256, 128)
        self.block9 = ResNetBlock(128) 
        self.cf2 = CompressFeatures(128, 64)
        self.block10 = ResNetBlock(64) 
        self.cf3 = CompressFeatures(64, 32)
        self.block11 = ResNetBlock(32)
        self.cf_final = CompressFeatures(32, output_channels)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x=self.cf1(x)
        x=self.up(x)
        x=self.block9(x)
        x=self.cf2(x)
        x=self.up(x)
        x=self.block10(x)
        x=self.cf3(x)
        x=self.up(x)
        x=self.block11(x)
        x=self.cf_final(x)
        x=self.sig(x)
        return x

