import torch.nn as nn


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

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        self.vertebrae = nn.ModuleList(
            [InvertedResidual(32, 32, stride=1, expand_ratio=6) for _ in range(self.num_blocks)])

        self.trunk_conv = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.upconv1 = nn.PixelShuffle(2)
        self.conv_exp1 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.PixelShuffle(2)

        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

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
        print(x.shape)
        x = self.conv_exp1(self.upconv1(x))
        print(x.shape)
        x = self.final_conv(self.upconv2(x))

        return x
