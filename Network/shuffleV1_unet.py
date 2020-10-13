import torch.nn as nn
import torch



def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class conv3x3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class basic_unit(nn.Module):
    '''
    shuffle basic unit
    '''
    def __init__(self, ch_in, ch_out, groups=4):
        super(basic_unit, self).__init__()
        self.bottleneck = ch_out // 4
        self.groups = groups
        self.gconv_1x1_a = nn.Conv2d(ch_in, self.bottleneck, kernel_size=1, groups=self.groups, bias=False)
        self.bn_1x1_a = nn.BatchNorm2d(self.bottleneck)
        self.relu_1x1_a = nn.ReLU(inplace=True)
        self.conv_dw = nn.Conv2d(self.bottleneck, self.bottleneck, kernel_size=3, stride=1, padding=1, groups=self.bottleneck, bias=False)
        self.bn_dw = nn.BatchNorm2d(self.bottleneck)
        self.gconv_1x1_b = nn.Conv2d(self.bottleneck, ch_out, kernel_size=1, groups=self.groups, bias=False)
        self.bn_1x1_b = nn.BatchNorm2d(ch_out)
        self.relu_1x1_b = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.gconv_1x1_a(x)
        out = self.bn_1x1_a(out)
        out = self.relu_1x1_a(out)
        out = channel_shuffle(out, self.groups)
        out = self.conv_dw(out)
        out = self.bn_dw(out)
        out = self.gconv_1x1_b(out)
        out = self.bn_1x1_b(out)
        out = self.relu_1x1_b(out)

        return out



class double_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(double_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        if self.ch_in == 1:
            self.conv = nn.Sequential(
                conv3x3(self.ch_in, self.ch_out),
                basic_unit(self.ch_out, self.ch_out, groups=4),
            )
        else:
            self.conv = nn.Sequential(
                basic_unit(ch_in, ch_out, groups=4),
                basic_unit(ch_out, ch_out, groups=4),
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
            basic_unit(ch_in, ch_out),
        )

    def forward(self, x):
        x = self.up(x)
        return x



class ShuffleV1_UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(ShuffleV1_UNet, self).__init__()

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