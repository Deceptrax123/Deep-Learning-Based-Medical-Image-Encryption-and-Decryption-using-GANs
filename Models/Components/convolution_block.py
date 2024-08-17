import torch
from torch import nn
from torch.nn import Conv2d, ConvTranspose2d, ReLU, InstanceNorm2d, Module, Identity


class ConvolutionBlock(Module):
    def __init__(self, in_channels, out_channels, is_downsampling=True, add_activation=True, **kwargs):

        super(ConvolutionBlock, self).__init__()

        if is_downsampling:
            self.conv = Conv2d(
                in_channels=in_channels, out_channels=out_channels, padding_mode='reflect', **kwargs)
            self.norm = InstanceNorm2d(out_channels)
            if add_activation:
                self.activation = ReLU(inplace=True)
            else:
                self.activation = Identity()
        else:
            self.conv = ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels, **kwargs)
            self.norm = InstanceNorm2d(out_channels)
            if add_activation:
                self.activation = ReLU(inplace=True)
            else:
                self.activation = Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        return x
