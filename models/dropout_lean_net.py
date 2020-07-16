import torch
import torch.nn as nn

from model_utils import *

class Encoder(nn.Module):
    def __init__(self, input_channels=4):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout3d()
        self.sig = nn.Sigmoid()
        self.initLayer = nn.Conv3d(input_channels, 32, 
                kernel_size=3, stride=1, padding=1)
        self.block0 = ResNetBlockWithDropout(32)
        self.ds1 = Downsample(32)
        self.block1 = ResNetBlockWithDropout(64)
        self.ds2 = Downsample(64)
        self.block3 = ResNetBlockWithDropout(128) 
        self.ds3 = Downsample(128)
        self.block5 = ResNetBlockWithDropout(256) 
        self.block6 = ResNetBlockWithDropout(256) 
        self.block7 = ResNetBlockWithDropout(256) 
        self.block8 = ResNetBlockWithDropout(256)

    def forward(self, x):
        # sp* is the state of the output at each spatial level
        sp0 = self.dropout(self.initLayer(x))
        sp1 = self.block0(sp0)
        sp2 = self.ds1(sp1)
        sp2 = self.block1(sp2)
        sp3 = self.ds2(sp2)
        sp3 = self.block3(sp3)
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
        self.block9 = ResNetBlockWithDropout(128) 
        self.cf2 = CompressFeatures(128, 64)
        self.block11 = ResNetBlockWithDropout(64) 
        self.cf3 = CompressFeatures(64, 32)
        self.block13 = ResNetBlockWithDropout(32)
        self.cf_final = CompressFeatures(32, output_channels)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.sig = nn.Sigmoid()

    def forward(self, x):
        sp3 = x['spatial_level_3'] + self.up(self.cf1(x['spatial_level_4']))
        sp3 = self.block9(sp3)
        sp2 = x['spatial_level_2'] + self.up(self.cf2(sp3))
        sp2 = self.block11(sp2)
        sp1 = x['spatial_level_1'] + self.up(self.cf3(sp2))
        sp1 = self.block13(sp1)
        output = self.sig(self.cf_final(sp1))

        return output
     

class DropoutLeaNet(nn.Module):
    def __init__(self):
        super(DropoutLeaNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
