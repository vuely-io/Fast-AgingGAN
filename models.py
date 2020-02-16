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
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
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

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        self.vertebrae = nn.ModuleList(
            [InvertedResidual(64, 64, stride=1, expand_ratio=6) for _ in range(self.num_blocks)])

        self.trunk_conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.upconv1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_exp1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_exp2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Args:
            x: Tensor, input image of shape: (B, C, H, W)
        """
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        for layer_num in range(self.num_blocks):
            x = self.vertebrae[layer_num](x)
        x = self.bn3(self.trunk_conv(x))
        x = self.conv_exp1(self.upconv1(x))
        x = self.conv_exp2(self.upconv2(x))
        x = self.final_conv(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(65, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128, eps=0.001, track_running_stats=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001, track_running_stats=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512, eps=0.001, track_running_stats=True)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)

    def forward(self, x, condition):
        """
        Args:
            x: Tensor, the input image tensor for the discriminator.
            condition: Tensor, the age conditionality.
        """
        x = self.lrelu(self.conv1(x))
        x = torch.cat((x, condition), 1)
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x


class AgeClassifier(nn.Module):
    def __init__(self):
        super(AgeClassifier, self).__init__()
        self.model = resnet18(pretrained=True, progress=True)
        self.model.fc = nn.Linear(512, 5, bias=True)

    def forward(self, x):
        """
        Args:
            x: Tesnor, the input to extract features from.
        """
        return self.model(x)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = resnet18(pretrained=True, progress=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, x):
        """
        Args:
            x: Tensor, the input to extract features from.
        """
        return self.model(x)
