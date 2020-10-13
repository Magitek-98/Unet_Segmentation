import torch.nn as nn
import torch



class conv_ds(nn.Module):
    '''
    depthwise separable convolutions
    '''
    def __init__(self, ch_in, ch_out):
        super(conv_ds, self).__init__()
        self.conv_dw = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=ch_in, bias=False)
        self.bn_dw = nn.BatchNorm2d(ch_in)
        self.relu_dw = nn.ReLU(inplace=True)
        self.conv_pw = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(ch_out)
        self.relu_pw = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        x = self.relu_dw(x)
        x = self.conv_pw(x)
        x = self.bn_pw(x)
        x = self.relu_pw(x)
        return x


class double_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            conv_ds(ch_in, ch_out),
            conv_ds(ch_out, ch_out),
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
            conv_ds(ch_in, ch_out),
        )

    def forward(self, x):
        x = self.up(x)
        return x



class Mobile_UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(Mobile_UNet, self).__init__()

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