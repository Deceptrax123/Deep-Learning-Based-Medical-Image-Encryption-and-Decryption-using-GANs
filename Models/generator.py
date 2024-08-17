import torch
from torch import nn
from torch.nn import Module, Conv2d, ReLU, InstanceNorm2d
from Models.Components.convolution_block import ConvolutionBlock
from Models.Components.residual_block import ResidualBlock


class Generator(Module):
    def __init__(self, img_channels: int, num_features: int = 64, num_residuals: int = 6):

        super(Generator, self).__init__()

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

        # Bottleneck
        self.residual_layers = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        # Decoder
        self.dec1 = ConvolutionBlock(
            num_features*4, num_features*2, is_downsampling=False, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = ConvolutionBlock(
            num_features*2, num_features, is_downsampling=False, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.final_layer = Conv2d(
            num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')

    def forward(self, x):
        x = self.inconv(x)
        x = self.innorm(x)
        x = self.activation(x)

        x = self.enc1(x)
        x = self.enc2(x)

        x = self.residual_layers(x)

        x = self.dec1(x)
        x = self.dec2(x)

        x = self.final_layer(x)

        return x
