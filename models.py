import functools

import torch
import torch.nn as nn
from torchvision.models import resnet18


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        """
        Args:
            inp: int, number of filters in the input feature map
            oup: int, number of filters in the output feature map
            stride: int, stride of the conv layers
            expand_ratio: expansion ratio before the depthwise conv
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        """
        Args:
            x: Tensor, The input feature map.
        """
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileGenerator(nn.Module):
    def __init__(self, num_blocks):
        """
        Args:
            num_blocks: int, number of blocks in the generator
        """
        super(MobileGenerator, self).__init__()
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)

        self.vertebrae = nn.ModuleList(
            [InvertedResidual(64, 64, stride=1, expand_ratio=6) for _ in range(self.num_blocks)])

        self.trunk_conv = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.upconv1 = nn.PixelShuffle(upscale_factor=2)
        self.conv_exp1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.PixelShuffle(upscale_factor=2)
        self.conv_exp2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Tensor, input image of shape: (B, C, H, W)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        feat = x.clone()
        for layer_num in range(self.num_blocks):
            x = self.vertebrae[layer_num](x)
        x += feat
        x = self.relu(self.bn3(self.trunk_conv(x)))
        x = self.relu(self.conv_exp1(self.upconv1(x)))
        x = self.relu(self.conv_exp2(self.upconv2(x)))
        x = self.sigmoid(self.final_conv(x))

        return x


class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        self.relu1 = nn.LeakyReLU(0.2, True)
        sequence = []
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if n == 1:
                cf = 1
            else:
                cf = 0
            sequence += [
                nn.Conv2d(cf + ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x, cond):
        x = self.relu1(self.conv1(x))
        x = torch.cat([x, cond], dim=1)
        return self.model(x)


class AgeClassifier(nn.Module):
    def __init__(self):
        super(AgeClassifier, self).__init__()
        self.model = resnet18(pretrained=True, progress=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(4 * 4 * 512, 5, bias=True)

    def forward(self, x):
        """
        Args:
            x: Tensor, the input to extract features from.
        Returns:
            x: The predicted class logits.
            features: The final conv features of the image.
        """
        b, _, _, _ = x.shape
        features = self.model(x)
        x = self.fc(self.pool(features).view(b, -1))
        return x, features
