import torch
import torch.nn as nn

from models.model_utils import *


class CoarseEncoder(nn.Module):
    def __init__(self, input_channels=4):
        super(CoarseEncoder, self).__init__()
        self.initLayer = nn.Conv3d(input_channels, 16,
                                   kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout3d()
        self.block0 = ResNetBlock(16)
        self.block1 = ResNetBlock(16)
        self.ds1 = Downsample(16)
        self.block2 = ResNetBlock(32)
        self.block3 = ResNetBlock(32)
        self.ds2 = Downsample(32)
        self.block4 = ResNetBlock(64)
        self.block5 = ResNetBlock(64)
        self.ds3 = Downsample(64)
        self.block6 = ResNetBlock(128)
        self.block7 = ResNetBlock(128)
        self.block8 = ResNetBlock(128)
        self.block9 = ResNetBlock(128)

    def forward(self, x):
        # sp* is the state of the output at each spatial level
        sp0 = self.dropout(self.initLayer(x))
        sp1 = self.block0(sp0)
        sp1 = self.block1(sp0)
        sp2 = self.ds1(sp1)
        sp2 = self.block2(sp2)
        sp2 = self.block3(sp2)
        sp3 = self.ds2(sp2)
        sp3 = self.block4(sp3)
        sp3 = self.block5(sp3)
        sp4 = self.ds3(sp3)
        sp4 = self.block6(sp4)
        sp4 = self.block7(sp4)
        sp4 = self.block8(sp4)
        sp4 = self.block9(sp4)

        return {
            'spatial_level_4': sp4, 'spatial_level_3': sp3,
            'spatial_level_2': sp2, 'spatial_level_1': sp1
        }


class CoarseDecoder(nn.Module):
    def __init__(self, output_channels=3):
        super(CoarseDecoder, self).__init__()
        self.block9 = ResNetBlock(64)
        self.block11 = ResNetBlock(32)
        self.block13 = ResNetBlock(16)

        self.up1 = UpsamplingDeconv3d(128, 64)
        self.up2 = UpsamplingDeconv3d(64, 32)
        self.up3 = UpsamplingDeconv3d(32, 16)

        self.cf_final = CompressFeatures(16, output_channels)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        sp3 = x['spatial_level_3'] + self.up1(x['spatial_level_4'])
        sp3 = self.block9(sp3)
        sp2 = x['spatial_level_2'] + self.up2(sp3)
        sp2 = self.block11(sp2)
        sp1 = x['spatial_level_1'] + self.up3(sp2)
        sp1 = self.block13(sp1)
        output = self.sig(self.cf_final(sp1))
        return output


class Encoder(nn.Module):
    def __init__(self, input_channels=4):
        super(Encoder, self).__init__()
        self.initLayer = nn.Conv3d(input_channels, 32,
                                   kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout3d()
        self.block0 = ResNetBlock(32)
        self.block1 = ResNetBlock(32)
        self.ds1 = Downsample(32)
        self.block2 = ResNetBlock(64)
        self.block3 = ResNetBlock(64)
        self.ds2 = Downsample(64)
        self.block4 = ResNetBlock(128)
        self.block5 = ResNetBlock(128)
        self.ds3 = Downsample(128)
        self.block6 = ResNetBlock(256)
        self.block7 = ResNetBlock(256)
        self.block8 = ResNetBlock(256)
        self.block9 = ResNetBlock(256)

    def forward(self, x):
        # sp* is the state of the output at each spatial level
        sp0 = self.dropout(self.initLayer(x))
        sp1 = self.block0(sp0)
        sp1 = self.block1(sp0)
        sp2 = self.ds1(sp1)
        sp2 = self.block2(sp2)
        sp2 = self.block3(sp2)
        sp3 = self.ds2(sp2)
        sp3 = self.block4(sp3)
        sp3 = self.block5(sp3)
        sp4 = self.ds3(sp3)
        sp4 = self.block6(sp4)
        sp4 = self.block7(sp4)
        sp4 = self.block8(sp4)
        sp4 = self.block9(sp4)

        return {
            'spatial_level_4': sp4, 'spatial_level_3': sp3,
            'spatial_level_2': sp2, 'spatial_level_1': sp1
        }


class BilineDecoder(nn.Module):
    def __init__(self, output_channels=3):
        super(BilineDecoder, self).__init__()
        self.cf1 = CompressFeatures(256, 128)
        self.block9 = ResNetBlock(128)
        self.cf2 = CompressFeatures(128, 64)
        self.block11 = ResNetBlock(64)
        self.cf3 = CompressFeatures(64, 32)
        self.block13 = ResNetBlock(32)
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


class DeconvDecoder(nn.Module):
    def __init__(self, output_channels=3):
        super(DeconvDecoder, self).__init__()
        self.block9 = ResNetBlock(128)
        self.block11 = ResNetBlock(64)
        self.block13 = ResNetBlock(32)
        self.cf_final = CompressFeatures(32, output_channels)

        self.up43 = UpsamplingDeconv3d(256, 128)
        self.up32 = UpsamplingDeconv3d(128, 64)
        self.up21 = UpsamplingDeconv3d(64, 32) 

        self.sig = nn.Sigmoid()

    def forward(self, x):
        sp3 = x['spatial_level_3'] + self.up43(x['spatial_level_4'])
        sp3 = self.block9(sp3)
        sp2 = x['spatial_level_2'] + self.up32(sp3)
        sp2 = self.block11(sp2)
        sp1 = x['spatial_level_1'] + self.up21(sp2)
        sp1 = self.block13(sp1)
        logits = self.cf_final(sp1)
        output = self.sig(logits)

        return output, logits


class CascadeNet(nn.Module):
    def __init__(self, lite=False):
        super(CascadeNet, self).__init__()
        self.lite = lite
        self.coarse_encoder = CoarseEncoder()
        self.coarse_decoder = CoarseDecoder()
        self.encoder = Encoder(input_channels=7)
        if not self.lite:
            self.deconv_decoder = DeconvDecoder()
        self.biline_decoder = BilineDecoder()

    def forward(self, x):
        # Uncomment these lines to use deconvolution
        coarse = self.coarse_encoder(x)
        coarse = self.coarse_decoder(coarse)

        x = torch.cat((coarse, x), 1) 
        x = self.encoder(x)
        deconv = None
        if not self.lite:
            deconv, deconv_logits = self.deconv_decoder(x)
        biline = self.biline_decoder(x)

        # Uncomment these lines for model that doesn't use
        # deconvolution.
        #x = self.encoder(x)
        #coarse = deconv = biline = self.biline_decoder(x)
        return {
            'coarse': coarse,
            'biline': biline,
            'deconv': deconv
        }, None 
