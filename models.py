import torch
import torch.nn as nn

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
            return out + residual
        return out

class DownSampling(nn.Module):
    # downsample by 2; simultaneously increase feature size by 2
    def __init__(self, in_channels):
        super(DownSampling, self).__init__()
        self.conv3x3x3 = nn.Conv3d(in_channels, 2*in_channels, 
            kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv3x3x3(x)

class UpsamplingBilinear3d(nn.modules.Upsample):
    def __init__(self, size=None, scale_factor=2):
        super(UpsamplingBilinear3d, self).__init__(size, scale_factor, 
                mode='trilinear', align_corners=True)

class CompressFeatures(nn.Module):
    # Reduce the number of features by a factor of 2.
    # Assumes channels_in is power of 2.
    def __init__(self, channels_in, channels_out):
        super(CompressFeatures, self).__init__()
        self.conv1x1x1 = nn.Conv3d(channels_in, channels_out,
                kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv1x1x1(x)

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

cfg = {
    'encoder': (4, 8, [
        32,
        "DS",
        64,
        64,
        "DS",
        128,
        128,
        "DS",
        256,
        256,
        256,
        256]
        ),
    'decoder': (256, 8, [
        "UP",
        "CF-128",
        128,
        128,
        "UP",
        "CF-64",
        64,
        64,
        "UP",
        "CF-32",
        32,
        32,
        "CF-3"]
        )
    }

def build_architecture(cfg, arc_type):
    in_channels, num_groups, architecture = cfg
    layers = nn.ModuleList()
    temp_layer = list()
    for layer in architecture:
        if layer == "DS":
            layers.append(nn.Sequential(*temp_layer))
            temp_layer = list()
            temp_layer += [DownSampling(in_channels)]
            in_channels*=2
        elif layer == "UP":
            temp_layer+=[UpsamplingBilinear3d()]
        elif "CF" in str(layer):
            _, out_channels = layer.split('-')
            out_channels = int(out_channels)
            temp_layer+=[CompressFeatures(in_channels, 
                out_channels)]
            in_channels = out_channels
            layers.append(nn.Sequential(*temp_layer))
            temp_layer = list()
        elif type(layer) == int:
            if temp_layer or arc_type=='decoder':
                temp_layer += [ResNetBlock(in_channels, layer, num_groups=num_groups)]
            else:
                # The first layer of the encoder does not use group norm and simply
                # expands the channel size.
                temp_layer += [ResNetBlock(in_channels, layer)]
            in_channels = layer
        else:
            raise ValueError(f'"{layer}" is not a valid layer specification.')
    if arc_type == 'decoder':
        temp_layer+=[nn.Sigmoid()]
    layers.append(nn.Sequential(*temp_layer))
    return layers

def build_net(cfg):
    return build_architecture(cfg['encoder'], 'encoder'),\
            build_architecture(cfg['decoder'], 'decoder')

class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        self.encoder, self.decoder = build_net(cfg)

    def forward(self, x):
        residuals = []
        for layer in self.encoder:
            x = layer(x)
            residuals.append(x)
        spatial_lvl = -2
        for layer in self.decoder:
            x = layer(x)
            if abs(spatial_lvl) <= len(residuals):
                x = x + residuals[spatial_lvl]
            spatial_lvl-=1
        return x
        
if __name__=='__main__':
    unet = UNet(cfg)
#def make_layers(cfg, batch_norm=False):
#    layers = list()
#    in_channels = 4
#    for v in cfg:
#        if v == "M":
#            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#        else:
#            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#            if batch_norm:
#                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#            else:
#                layers += [conv2d, nn.ReLU(inplace=True)]
#            in_channels = v
#    return nn.Sequential(*layers)
#
#cfg = {
#    16: [
#        64,
#        64,
#        "M",
#        128,
#        128,
#        "M",
#        256,
#        256,
#        256,
#        "M",
#        512,
#        512,
#        512,
#        "M",
#        512,
#        512,
#        512,
#        "M",
#    ],

