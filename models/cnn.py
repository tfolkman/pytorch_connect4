import torch
from torch import nn


class IdentityBlock(nn.Module):
    def __init__(self, in_channel, out_channels, f, in_size, conv_block=False,
                 s=2):
        super(IdentityBlock, self).__init__()
        if conv_block:
            self.conv1 = nn.Conv2d(in_channel, out_channels[0], 1, s)
            out1_size = int(((in_size - 1) / s) + 1)
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channels[0], 1)
            out1_size = int(((in_size - 1) / 1) + 1)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        ## we want same padding here
        conv2_padding = int((f - 1) / 2)
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], f,
                               padding=conv2_padding)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], 1)
        self.bn3 = nn.BatchNorm2d(out_channels[2])

        # if conv block need some additional layers
        self.conv_block = conv_block
        if self.conv_block:
            self.shortcut_conv = nn.Conv2d(in_channel, out_channels[2], 1, stride=s)
            self.shortcut_bn = nn.BatchNorm2d(out_channels[2])

    def forward(self, x):
        conv1_out = functional.relu(self.bn1(self.conv1(x)))
        conv2_out = functional.relu(self.bn2(self.conv2(conv1_out)))
        conv3_out = self.bn3(self.conv3(conv2_out))
        if self.conv_block:
            shortcut = self.shortcut_bn(self.shortcut_conv(x))
        else:
            shortcut = x
        return functional.relu(shortcut + conv3_out)

class Residual_CNN(nn.Module):
    def __init__(self):
        super(Residual_CNN, self).__init__()