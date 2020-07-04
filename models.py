import torch
import torch.nn as nn

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
    def __init__(self, in_channels, out_channels, num_groups=None):
        super(ResNetBlock, self).__init__()
        if num_groups:
            self.feats = nn.Sequential(nn.GroupNorm(num_groups, in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, out_channels, 
                    kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups, out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, 
                    kernel_size=3, stride=1, padding=1))
        else:
            self.feats = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 
                    kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True))

    def forward(self, x):
        residual = x
        out = self.feats(x)
        
        if out.size() == residual.size():
            out += residual
        return out

# I don't love the next two classes. They're just thin wrappers over nn.Con3d.
class Downsample(nn.Module):
    # downsample by 2; simultaneously increase feature size by 2
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.conv3x3x3 = nn.Conv3d(in_channels, 2*in_channels, 
            kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv3x3x3(x)


# TODO: maybe get rid of this? it's just thin wrapper around a
# convolutional layer.
class CompressFeatures(nn.Module):
    # Reduce the number of features by a factor of 2.
    # Assumes channels_in is power of 2.
    def __init__(self, channels_in, channels_out):
        super(CompressFeatures, self).__init__()
        self.conv1x1x1 = nn.Conv3d(channels_in, channels_out,
                kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv1x1x1(x)

# TODO: I don't see the use in this thin wrapper over ConvTranspose3d.
class UpsamplingDeconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4):
        super(UpsamplingDeconv3d, self).__init__()
        self.deconv = torch.nn.ConvTranspose3d(in_channels,
                out_channels,
                kernel_size,
                stride=2,
                padding=1,
                output_padding=0,
                groups=1,
                bias=True,
                dilation=1,
                padding_mode='zeros')

        def forward(self, x):
            return self.deconv(x)


class Encoder(nn.Module):
    def __init__(self, dropout, input_channels=4):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.sig = nn.Sigmoid()
        self.initLayer = nn.Conv3d(input_channels, 32, 
                kernel_size=3, stride=1, padding=1)
        self.block0 = SimpleResNetBlock(32)
        self.ds1 = Downsample(32)
        self.block1 = SimpleResNetBlock(64)
        self.block2 = SimpleResNetBlock(64) 
        self.ds2 = Downsample(64)
        self.block3 = SimpleResNetBlock(128) 
        self.block4 = SimpleResNetBlock(128) 
        self.ds3 = Downsample(128)
        self.block5 = SimpleResNetBlock(256) 
        self.block6 = SimpleResNetBlock(256) 
        self.block7 = SimpleResNetBlock(256) 
        self.block8 = SimpleResNetBlock(256)

    def forward(self, x):
        # sp* is the state of the output at each spatial level
        sp0 = self.initLayer(x)
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
    def __init__(self, dropout, output_channels=3):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.cf1 = CompressFeatures(256, 128)
        self.block9 = SimpleResNetBlock(128) 
        self.block10 = SimpleResNetBlock(128) 
        self.cf2 = CompressFeatures(128, 64)
        self.block11 = SimpleResNetBlock(64) 
        self.block12 = SimpleResNetBlock(64) 
        self.cf3 = CompressFeatures(64, 32)
        self.block13 = SimpleResNetBlock(32)
        self.block14 = SimpleResNetBlock(32)
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
    def __init__(self, dropout=True):
        super(MonoUNet, self).__init__()
        self.encoder = Encoder(dropout)
        self.decoder = Decoder(dropout)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
