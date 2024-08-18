import torch
from torch import nn
from torch.nn import Module, Conv2d, ReLU, InstanceNorm2d
from Models.Components.convolution_block import ConvolutionBlock


class Encryption(Module):
    def __init__(self, img_channels: int, num_features: int = 64):

        super(Encryption, self).__init__()

        # Initials
        self.inconv = Conv2d(img_channels, num_features, kernel_size=7,
                             stride=1, padding=3, padding_mode='reflect')
        self.innorm = InstanceNorm2d(num_features)
        self.activation = ReLU(inplace=True)

        # Encoder
        self.enc1 = ConvolutionBlock(
            num_features, num_features*2, is_downsampling=True, kernel_size=3, stride=2, padding=1)
        self.enc2 = ConvolutionBlock(
            num_features*2, num_features*4, is_downsampling=True, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.inconv(x)
        x = self.innorm(x)
        x = self.activation(x)

        x = self.enc1(x)
        x = self.enc2(x)

        return x
