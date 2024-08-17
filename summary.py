from Models.generator import Generator
from Models.discriminator import Discriminator
from torchsummary import summary

# Generator Summary
model = Generator(img_channels=3)
summary(model, input_size=(1, 256, 256), batch_size=-1, device='cpu')

# Discriminator Summary
model = Discriminator(in_channels=3)
summary(model, input_size=(1, 256, 256), batch_size=-1, device='cpu')
