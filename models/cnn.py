from torch.nn import functional
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
        # we want same padding here
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


class ResNet50Extractor(nn.Module):
    """
    Takes in an data and returns a 2048 vector using residual layers
    """
    def __init__(self, in_channels):
        super(ResNet50Extractor, self).__init__()

        # stage 1
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d(3, stride=2)

        # stage 2
        self.conv_block1 = IdentityBlock(64, [64, 64, 256], 3, 15,
                                         conv_block=True, s=1)
        self.id_block1 = IdentityBlock(256, [64, 64, 256], 3, 15)
        self.id_block2 = IdentityBlock(256, [64, 64, 256], 3, 15)

        # stage 3
        self.conv_block2 = IdentityBlock(256, [128, 128, 512], 3, 15,
                                         conv_block=True, s=2)
        self.id_block3 = IdentityBlock(512, [128, 128, 512], 3, 15)
        self.id_block4 = IdentityBlock(512, [128, 128, 512], 3, 15)
        self.id_block5 = IdentityBlock(512, [128, 128, 512], 3, 15)

        # stage 4
        self.conv_block3 = IdentityBlock(512, [256, 256, 1024], 3, 8,
                                         conv_block=True, s=2)
        self.id_block6 = IdentityBlock(1024, [256, 256, 1024], 3, 8)
        self.id_block7 = IdentityBlock(1024, [256, 256, 1024], 3, 8)
        self.id_block8 = IdentityBlock(1024, [256, 256, 1024], 3, 8)
        self.id_block9 = IdentityBlock(1024, [256, 256, 1024], 3, 8)
        self.id_block10 = IdentityBlock(1024, [256, 256, 1024], 3, 8)

        # stage 5
        self.conv_block4 = IdentityBlock(1024, [512, 512, 2048], 3, 4,
                                         conv_block=True, s=2)
        self.id_block11 = IdentityBlock(2048, [512, 512, 2048], 3, 4)
        self.id_block12 = IdentityBlock(2048, [512, 512, 2048], 3, 4)

    def forward(self, x):
        x = self.mp1(functional.relu(self.bn1(self.conv1(x))))
        x = self.conv_block1(x)
        x = self.id_block2(self.id_block1(x))
        x = self.id_block5(self.id_block4(self.id_block3(self.conv_block2(x))))
        x = self.id_block8(self.id_block7(self.id_block6(self.conv_block3(x))))
        x = self.id_block10(self.id_block9(x))
        x = self.id_block12(self.id_block11(self.conv_block4(x)))
        return x


class LinearHead(nn.Module):
    def __init__(self, in_channels, out_channels, final_out, tanh_activate):
        super(LinearHead, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.linear = nn.Linear(out_channels, final_out, bias=False)
        self.tahn_activate = tanh_activate

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn(x)
        x = functional.leaky_relu(x)
        x = self.linear(x)
        if self.tahn_activate:
            return functional.tanh(x)
        else:
            return x


class ValuePolicyModel(nn.Module):
    def __init__(self, in_channels, action_size):
        super(ValuePolicyModel, self).__init__()

        self.res_layers = ResNet50Extractor(in_channels)
        self.value_head = LinearHead(2048, 1, 1, True)
        self.policy_head = LinearHead(2048, 2, action_size, False)

    def forward(self, x):
        res_features = self.res_layers(x)
        return self.value_head(res_features), self.policy_head(res_features)
