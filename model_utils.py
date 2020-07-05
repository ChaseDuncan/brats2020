import torch
import torch.nn as nn


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

class ResNetBlockWithDropout(nn.Module):
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
    def __init__(self, channels, num_groups=32):
        super(SimpleResNetBlock, self).__init__()
        self.feats = nn.Sequential(nn.GroupNorm(num_groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 
                kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 
                kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        residual = x
        out = self.feats(x)
        out += residual
        return out


