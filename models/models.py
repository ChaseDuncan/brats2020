import torch
import torch.nn as nn

from .model_utils import *
from .cascade_net import DeconvDecoder

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


class Encoder(nn.Module):
    def __init__(self, input_channels=4, instance_norm=False):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout3d(p=0.2)
        self.sig = nn.Sigmoid()
        self.initLayer = nn.Conv3d(input_channels, 32, 
                kernel_size=3, stride=1, padding=1)
        self.block0 = ResNetBlock(32, instance_norm=instance_norm)
        self.ds1 = Downsample(32)
        self.block1 = ResNetBlock(64, instance_norm=instance_norm)
        self.block2 = ResNetBlock(64, instance_norm=instance_norm) 
        self.ds2 = Downsample(64)
        self.block3 = ResNetBlock(128, instance_norm=instance_norm) 
        self.block4 = ResNetBlock(128, instance_norm=instance_norm) 
        self.ds3 = Downsample(128)
        self.block5 = ResNetBlock(256, instance_norm=instance_norm) 
        self.block6 = ResNetBlock(256, instance_norm=instance_norm) 
        self.block7 = ResNetBlock(256, instance_norm=instance_norm) 
        self.block8 = ResNetBlock(256, instance_norm=instance_norm)

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
    def __init__(self, output_channels=3, instance_norm=False):
        super(Decoder, self).__init__()
        self.cf1 = CompressFeatures(256, 128)
        self.block9 = ResNetBlock(128, instance_norm=instance_norm) 
        self.block10 = ResNetBlock(128, instance_norm=instance_norm) 
        self.cf2 = CompressFeatures(128, 64)
        self.block11 = ResNetBlock(64, instance_norm=instance_norm) 
        self.block12 = ResNetBlock(64, instance_norm=instance_norm) 
        self.cf3 = CompressFeatures(64, 32)
        self.block13 = ResNetBlock(32, instance_norm=instance_norm)
        self.block14 = ResNetBlock(32, instance_norm=instance_norm)
        self.cf_final = CompressFeatures(32, output_channels)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.sig = nn.Sigmoid()

    def forward(self, x):
        sp3 = x['spatial_level_3'] + self.up(self.cf1(x['spatial_level_4']))
        sp3 = self.block9(sp3)
        #sp3 = self.block10(sp3)
        sp2 = x['spatial_level_2'] + self.up(self.cf2(sp3))
        sp2 = self.block11(sp2)
        #sp2 = self.block12(sp2)
        sp1 = x['spatial_level_1'] + self.up(self.cf3(sp2))
        sp1 = self.block13(sp1)
        #sp1 = self.block14(sp1)
        logits = self.cf_final(sp1)
        return self.sig(logits), logits
     

class MonoUNet(nn.Module):
    def __init__(self, input_channels=4, upsampling='bilinear', instance_norm=False):
        super(MonoUNet, self).__init__()
        self.encoder = Encoder(input_channels=input_channels, instance_norm=instance_norm)
        if upsampling == 'deconv':
            self.decoder = DeconvDecoder()
        else:
            self.decoder = Decoder(instance_norm=instance_norm)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class HierarchicalNet(nn.Module):
    def __init__(self, cp1, cp2, device):
        super(HierarchicalNet, self).__init__()
        self.model1 = MonoUNet()
        self.model2 = MonoUNet(input_channels=5)

        checkpoint1 = torch.load(cp1, map_location=device)
        self.model1.load_state_dict(checkpoint1['state_dict'], strict=False)
        checkpoint2 = torch.load(cp2, map_location=device)
        self.model2.load_state_dict(checkpoint2['state_dict'], strict=False)

    def forward(self, x):
        output, _ = self.model1(x)
        x = torch.cat((x, output[:, 1, :, :, :].unsqueeze(1)), 1)
        output2, _ = self.model2(x)
        output2[:, 1, :, :, :] = output[:, 1, :, :, :]
        output = output2
        return output, None

