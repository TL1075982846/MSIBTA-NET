import torch.nn as nn
import torch

from triplet_attention import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

        self.triplet_attention = TripletAttention(out_channel, 16)


    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        out = self.triplet_attention(out)

        out += identity
        out = self.relu(out)

        return out


'''class Bottleneck1(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64,):
        super(Bottleneck1, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        # -----------------------------------------

        self.conv1_1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.conv1_2 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel * self.expansion)
        # -----------------------------------------

        self.conv2_1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                                 kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.conv2_2 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion, groups=groups,
                               kernel_size=5, stride=stride, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channel * self.expansion)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=stride, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.triplet_attention = TripletAttention(out_channel * 4, 16)


    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)





        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out1 = self.conv1_1(x)
        out1 = self.conv1_2(out1)
        out1 = self.bn1(out1)
        out2 = self.conv2_1(x)
        out2 = self.conv2_2(out2)
        out2 = self.bn2(out2)
        out3 = self.conv3(x)
        out3 = self.bn3(out3)
        out = out1 + out2 + out3

        if self.downsample is not None:
            residual = self.downsample(x)


        out = self.triplet_attention(out)

        out += identity
        out = self.relu(out)

        return out'''

class Bottleneck1(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck1, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1_1 = nn.Conv2d(in_channels=in_channel, out_channels=width*2,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width*2)
        # -----------------------------------------

        self.conv1_2 = nn.Conv2d(in_channels=width*2, out_channels=width*6, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width*6)
        # -----------------------------------------
        self.conv1_3 = nn.Conv2d(in_channels=width*6, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(in_channels=in_channel,out_channels=width*2,
                                 kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn4 = nn.BatchNorm2d(width*2)
        # -----------------------------------------

        self.conv2_2 = nn.Conv2d(in_channels=width*2, out_channels=width*6, groups=groups,
                                 kernel_size=5, stride=stride, bias=False, padding=2)
        self.bn5 = nn.BatchNorm2d(width*6)
        # -----------------------------------------
        self.conv2_3 = nn.Conv2d(in_channels=width*6, out_channels=out_channel*self.expansion,
                                 kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn6 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        '''self.conv3_1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                                 kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn7 = nn.BatchNorm2d(width)
        # -----------------------------------------

        self.conv3_2 = nn.Conv2d(in_channels=width, out_channels=width*4, groups=groups,
                                 kernel_size=5, stride=stride, bias=False, padding=2)
        self.bn8 = nn.BatchNorm2d(width*4)
        # -----------------------------------------
        self.conv3_3 = nn.Conv2d(in_channels=width*4, out_channels=out_channel*self.expansion,
                                 kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn9 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)'''

        self.downsample = downsample
        self.triplet_attention = TripletAttention(out_channel, 16)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out1_1 = self.conv1_1(x)
        out1_1 = self.bn1(out1_1)
        out1_1 = self.relu(out1_1)

        out1_2 = self.conv1_2(out1_1)
        out1_2 = self.bn2(out1_2)
        out1_2 = self.relu(out1_2)

        out1_3 = self.conv1_3(out1_2)
        out1 = self.bn3(out1_3)

        out2_1 = self.conv2_1(x)
        out2_1 = self.bn4(out2_1)
        out2_1 = self.relu(out2_1)

        out2_2 = self.conv2_2(out2_1)
        out2_2 = self.bn5(out2_2)
        out2_2 = self.relu(out2_2)

        out2_3 = self.conv2_3(out2_2)
        out2 = self.bn6(out2_3)

        '''out3_1 = self.conv3_1(x)
        out3_1 = self.bn4(out3_1)
        out3_1 = self.relu(out3_1)

        out3_2 = self.conv3_2(out3_1)
        out3_2 = self.bn5(out3_2)
        out3_2 = self.relu(out3_2)

        out3_3 = self.conv2_3(out3_2)
        out3 = self.bn6(out3_3)'''

        out = out1 + out2

        outa = self.triplet_attention(out)

        outa += identity
        outa = self.relu(outa)

        return outa

class Bottleneck2(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck2, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------

        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width*6, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width*6)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width*6, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.triplet_attention = TripletAttention(out_channel, 16)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        out3 = self.conv3(out2)
        out3 = self.bn3(out3)

        out4 = self.triplet_attention(out3)
        out4 += identity
        out4 = self.relu(out4)

        return out4


class ResNet(nn.Module):

    def __init__(self,
                 block1,
                 block2,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        #self.conv1 = nn.Conv2d(3,self.in_channel, kernel_size=1, stride=2,
        #                     bias=False)

        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        '''self.conv1_2 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)'''
        self.conv2_1 = nn.Conv2d(3,self.in_channel, kernel_size=1, stride=1,
                                  bias=False)
        self.conv2_2 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=5, stride=2,
                               padding=2, bias=False)
        self.conv3_1 = nn.Conv2d(3, self.in_channel, kernel_size=1, stride=1,
                                bias=False)
        self.conv3_2 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block1, 64, blocks_num[0])
        self.layer2 = self._make_layer(block2, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block1, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block2, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block1.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group
                                ))

        return nn.Sequential(*layers)

    def forward(self, x):
        #x1 = self.conv1(x)
        #x1 = self.maxpool(x1)
        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        x3 = self.conv3_1(x)
        x3 = self.conv3_2(x3)
        #x = x1 + x2 + x3
        x =  x2 + x3
        xm = self.bn1(x)
        xn = self.relu(xm)
        xl = self.maxpool(xn)

        xa = self.layer1(xl)
        xb = self.layer2(xa)
        xc = self.layer3(xb)
        xd = self.layer4(xc)

        if self.include_top:
            xe = self.avgpool(xd)
            xf = torch.flatten(xe, 1)
            xg = self.fc(xf)

        return xg


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=True)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck1,Bottleneck2, [2, 3, 3, 2], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck2, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck2, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck2, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

