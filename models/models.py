import torch
import torch.nn as nn

from model_utils import *

class SimpleResNetBlock(nn.Module):
    def __init__(self, channels, num_groups=32):
        super(SimpleResNetBlock, self).__init__()
        self.feats = nn.Sequential(nn.GroupNorm(num_groups, channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(),
            nn.Conv3d(channels, channels, 
                kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(),
            nn.Conv3d(channels, channels, 
                kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        residual = x
        out = self.feats(x)
        out += residual
        return out


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, num_groups=8):
        super(ResNetBlock, self).__init__()
        self.feats = nn.Sequential(nn.GroupNorm(num_groups, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, 
                kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, 
                kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        residual = x
        out = self.feats(x)
        out += residual
        return out

class Encoder(nn.Module):
    def __init__(self, input_channels=4):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout3d()
        self.sig = nn.Sigmoid()
        self.initLayer = nn.Conv3d(input_channels, 32, 
                kernel_size=3, stride=1, padding=1)
        self.block0 = ResNetBlock(32)
        self.ds1 = Downsample(32)
        self.block1 = ResNetBlock(64)
        self.block2 = ResNetBlock(64) 
        self.ds2 = Downsample(64)
        self.block3 = ResNetBlock(128) 
        self.block4 = ResNetBlock(128) 
        self.ds3 = Downsample(128)
        self.block5 = ResNetBlock(256) 
        self.block6 = ResNetBlock(256) 
        self.block7 = ResNetBlock(256) 
        self.block8 = ResNetBlock(256)

    def forward(self, x):
        # sp* is the state of the output at each spatial level
        sp0 = self.dropout(self.initLayer(x))
        sp1 = self.block0(sp0)
        sp2 = self.ds1(sp1)
        sp2 = self.block1(sp2)
        sp2 = self.block2(sp2)
        sp3 = self.ds2(sp2)
        sp3 = self.block3(sp3)
        sp3 = self.block4(sp3)
        sp4 = self.ds3(sp3)
        sp4 = self.block5(sp4)
        sp4 = self.block6(sp4)
        sp4 = self.block7(sp4)
        sp4 = self.block8(sp4)

        return {
                'spatial_level_4':sp4, 'spatial_level_3':sp3, 
                'spatial_level_2':sp2, 'spatial_level_1':sp1
                }


class Decoder(nn.Module):
    def __init__(self, output_channels=3):
        super(Decoder, self).__init__()
        self.cf1 = CompressFeatures(256, 128)
        self.block9 = ResNetBlock(128) 
        self.block10 = ResNetBlock(128) 
        self.cf2 = CompressFeatures(128, 64)
        self.block11 = ResNetBlock(64) 
        self.block12 = ResNetBlock(64) 
        self.cf3 = CompressFeatures(64, 32)
        self.block13 = ResNetBlock(32)
        self.block14 = ResNetBlock(32)
        self.cf_final = CompressFeatures(32, output_channels)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.sig = nn.Sigmoid()

    def forward(self, x):
        sp3 = x['spatial_level_3'] + self.up(self.cf1(x['spatial_level_4']))
        sp3 = self.block9(sp3)
        sp3 = self.block10(sp3)
        sp2 = x['spatial_level_2'] + self.up(self.cf2(sp3))
        sp2 = self.block11(sp2)
        sp2 = self.block12(sp2)
        sp1 = x['spatial_level_1'] + self.up(self.cf3(sp2))
        sp1 = self.block13(sp1)
        sp1 = self.block14(sp1)
        output = self.sig(self.cf_final(sp1))

        return output
     

class MonoUNet(nn.Module):
    def __init__(self):
        super(MonoUNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
