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

class DownSampling(nn.Module):
    # downsample by 2; simultaneously increase feature size by 2
    def __init__(self, in_channels):
        super(DownSampling, self).__init__()
        self.conv3x3x3 = nn.Conv3d(in_channels, 2*in_channels, 
            kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv3x3x3(x)

# TODO: get rid of this
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
        # TODO: don't do this; reverse list
        spatial_lvl = -2
        for layer in self.decoder:
            x = layer(x)
            if abs(spatial_lvl) <= len(residuals):
                x = x + residuals[spatial_lvl]
            spatial_lvl-=1
        return x


class Encoder(nn.Module):
    def __init__(self, input_channels=4):
        super(Encoder, self).__init__()
        self.sig = nn.Sigmoid()
        self.initLayer = nn.Conv3d(input_channels, 32, 
                kernel_size=3, stride=1, padding=1)
        self.block0 = SimpleResNetBlock(32)
        self.ds1 = DownSampling(32)
        self.block1 = SimpleResNetBlock(64)
        self.block2 = SimpleResNetBlock(64) 
        self.ds2 = DownSampling(64)
        self.block3 = SimpleResNetBlock(128) 
        self.block4 = SimpleResNetBlock(128) 
        self.ds3 = DownSampling(128)
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
    def __init__(self, output_channels=3):
        super(Decoder, self).__init__()
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
        #print([v.size() for k, v in x.items()])
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
