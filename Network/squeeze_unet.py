import torch.nn as nn
import math
import torch



class fire_module(nn.Module):
    '''
    SqueezeNet --fire module
    '''
    def __init__(self, ch_in, ch_out):
        super(fire_module, self).__init__()
        inplanes = ch_in # 1
        squeeze_planes = ch_out//8 # 8
        expand_planes = ch_out//2 # 32
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class double_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            fire_module(ch_in, ch_out),
            fire_module(ch_out, ch_out),
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class up_conv(nn.Module):
    '''
    两种方式实现
    1.反卷积
    2.上采样+卷积
    '''
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            fire_module(ch_in, ch_out),
        )

    def forward(self, x):
        x = self.up(x)
        return x



class Squeeze_UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(Squeeze_UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = double_conv(ch_in=1, ch_out=64)
        self.Conv2 = double_conv(ch_in=64, ch_out=128)
        self.Conv3 = double_conv(ch_in=128, ch_out=256)
        self.Conv4 = double_conv(ch_in=256, ch_out=512)
        self.Conv5 = double_conv(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = double_conv(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = double_conv(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = double_conv(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = double_conv(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1