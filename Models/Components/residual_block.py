import torch
from torch.nn import Module
from Models.Components.convolution_block import ConvolutionBlock


class ResidualBlock(Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.bl1 = ConvolutionBlock(
            channels, channels, add_activation=True, kernel_size=3, padding=1)
        self.bl2 = ConvolutionBlock(
            channels, channels, add_activation=False, kernel_size=3, padding=1)

    def forward(self, x):

        x1 = self.bl1(x)
        x2 = self.bl2(x1)

        return x2+x
