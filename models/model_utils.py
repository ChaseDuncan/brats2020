import os
import random
import torch
import torch.nn as nn


def model_average(model_dir, model, device, sample_proportion=0.33, sample_rate=0.5):
    model_cps = []
    for p, _, files in os.walk(f'{model_dir}/checkpoints/'):
        p_idx = int(sample_proportion*len(files))
        r_idx = int(sample_rate*p_idx)

        files = files[-p_idx:]
        random.shuffle(files)
        files = files[:r_idx]

        for f in files:
            model_cps.append(os.path.join(p, f))
    checkpoint_file = model_cps[-1] 
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    avg_params = dict(model.named_parameters())
    
    for i, checkpoint_file in enumerate(model_cps[:-1]):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        params = model.named_parameters()
        for name, param in params:
            if name in avg_params:
                avg_params[name].data.copy_((1/(i+1)*param.data + (i/(i+1))*avg_params[name].data))
    
    model.load_state_dict(avg_params)  
    return model


def gn_update(loader, model, device=None):
    r"""Updates GroupNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for GroupNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data group should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update GroupNorm
            statistics.
        device (torch.device, optional): If set, data will be trasferred to
            :attr:`device` before being passed into :attr:`model`.
    """
    #if not _check_gn(model):
    #    print('whoops')
    #    return
    was_training = model.training
    model.train()
    momenta = {}
    model.apply(_reset_gn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        b = input.size(0)

    momentum = b / float(n + b)
    for module in momenta.keys():
        module.momentum = momentum

    if device is not None:
        input = input.to(device)

    model(input)
    n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    model.train(was_training)


# GroupNorm utils
def _check_gn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.groupnorm._GroupNorm):
        flag[0] = True


def _check_gn(model):
    flag = [False]
    model.apply(lambda module: _check_gn_apply(module, flag))
    return flag[0]


def _reset_gn(module):
    if issubclass(module.__class__, torch.nn.modules.groupnorm._GroupNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.groupnorm._GroupNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.groupnorm._GroupNorm):
        module.momentum = momenta[module]

# egn gn_updata


def bn_update(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be trasferred to
            :attr:`device` before being passed into :attr:`model`.
    """
    if not _check_bn(model):
        print('whoops')
        return
    was_training = model.training
    model.train()
    momenta = {}
    model.apply(_reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        b = input.size(0)

    momentum = b / float(n + b)
    for module in momenta.keys():
        module.momentum = momentum

    if device is not None:
        input = input.to(device)

    model(input)
    n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    model.train(was_training)


# BatchNorm utils
def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

# ebn bn_updata


# I don't love the next two classes. They're just thin wrappers over nn.Con3d.
class Downsample(nn.Module):
    # downsample by 2; simultaneously increase feature size by 2
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.conv3x3x3 = nn.Conv3d(in_channels, 2*in_channels, 
            kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv3x3x3(x)


# maybe get rid of this? it's just thin wrapper around a convolutional layer.
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
    def __init__(self, channels, num_groups=8):
        super(ResNetBlockWithDropout, self).__init__()
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
    def __init__(self, channels, num_groups=8, instance_norm=False):
        super(ResNetBlock, self).__init__()
        if instance_norm:
            self.feats = nn.Sequential(nn.InstanceNorm3d(channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, 
                    kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, 
                    kernel_size=3, stride=1, padding=1))
        else:
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


# TODO
#class MultiResNetBlock(nn.Module):
#    def __init__(self, channels, num_groups=8):
#        super(MultiResNetBlock, self).__init__()
#
#            self.act = nn.ReLU(inplace=True),
#            self.gn1 = nn.GroupNorm(num_groups, channels),
#            self.cn1 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1),
#                nn.GroupNorm(num_groups, channels),
#                nn.ReLU(inplace=True),
#                nn.Conv3d(channels, channels, 
#                    kernel_size=3, stride=1, padding=1))
#
#    def forward(self, x):
#        residual = x
#        out = self.feats(x)
#        out += residual
#        return out

