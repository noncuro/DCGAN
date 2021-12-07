__all__ = ['Generator', 'Discriminator', 'weights_init']

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, groups=1, block=nn.Conv2d, relu_alpha=0.01, **kwargs):
        super(ResidualBlock, self).__init__()
        self.relu_alpha = relu_alpha
        self.left = nn.Sequential(
            block(inchannel, outchannel, kernel_size=3,
                  stride=stride, padding=1, bias=False, groups=groups, **kwargs),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(relu_alpha, inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3,
                      stride=1, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(outchannel))

        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                block(inchannel, outchannel, groups=groups,
                      kernel_size=1, stride=stride,
                      padding=0, bias=False, **kwargs),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.leaky_relu_(out, self.relu_alpha)
        return out


def TransposeBlock(**kwargs):
    return ResidualBlock(block=nn.ConvTranspose2d, **kwargs)


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        _leaky_alpha = 0.01
        self.initial_layers = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(_leaky_alpha),
        )
        self.layers = nn.Sequential(
            ResidualBlock(512, 512, relu_alpha=_leaky_alpha),
            ResidualBlock(512, 512, relu_alpha=_leaky_alpha),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(_leaky_alpha),
            ResidualBlock(256, 256, relu_alpha=_leaky_alpha),
            ResidualBlock(256, 256, relu_alpha=_leaky_alpha),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(_leaky_alpha),
            ResidualBlock(128, 128, relu_alpha=_leaky_alpha),
            ResidualBlock(128, 128, relu_alpha=_leaky_alpha),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(64, 64, relu_alpha=_leaky_alpha),
            ResidualBlock(64, 64, relu_alpha=_leaky_alpha),
            ResidualBlock(64, 3, relu_alpha=_leaky_alpha),
            nn.Tanh())

    def decode(self, x):
        x = self.initial_layers(x)
        x = self.layers(x)
        return x

    def forward(self, z: torch.Tensor):
        z = z.view(-1, self.latent_dim, 1, 1)
        return self.decode(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            ResidualBlock(3, 32, stride=1),
            ResidualBlock(32, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 128, stride=2),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 1, kernel_size=2),
            # NO SIGMOID - make sure loss function is appropriate
            # nn.Sigmoid()
        )

    def discriminator(self, x):
        out = self.layers(x)
        return out

    def forward(self, x):
        out = self.discriminator(x)
        return out.view(-1, 1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
