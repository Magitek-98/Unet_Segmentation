import torch.nn as nn
import torch



def channel_interleaved(x, M):
    '''
    承接上个卷积的操作：把输入通道划分为L个组，每个组包含M个通道，然后将L个分组卷积的结果拼接在一起得到新的输入其仍然有LxM个通道
    通道交错操作，划分了M个组，每个组包含L个通道，且这L个通道来自第一次分组卷积时的不同组。
    论文原文在M=2时效果达到最佳，但是我在实践过程中发现M=2时会极大的增加运行时间，经过计算后，这个网络已经改良为最小的分组。
    :param x:
    :param groups:
    :return:
    '''
    batchsize, num_channels, height, width = x.data.size()
    L = num_channels // M
    # reshape
    x = x.view(batchsize, L,
               M, height, width)

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

class IGCV_block(nn.Module):
    def __init__(self, ch_in, ch_out, M):
        '''
        :param ch_in:
        :param ch_out:
        :param L: 第一次组卷积中分组的数量（也是第二次组卷积中每个分组内包含的通道数）
        :param M: 第一次组卷积中每个分组内包含的通道数（也是第二次组卷积中的分组数量）
        '''
        super(IGCV_block, self).__init__()
        self.M = M
        self.gconv1 = nn.Conv2d(ch_in, ch_in, kernel_size=3, groups=ch_in//M, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_in)
        self.relu1 = nn.ReLU(inplace=True)
        self.gconv2 = nn.Conv2d(ch_in, ch_out, kernel_size=1, groups=M, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.gconv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = channel_interleaved(out, self.M)
        out = self.gconv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out



class double_conv(nn.Module):
    def __init__(self, ch_in, ch_out, M):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            IGCV_block(ch_in, ch_out, M),
            IGCV_block(ch_out, ch_out, M),
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
    def __init__(self, ch_in, ch_out, M):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            IGCV_block(ch_in, ch_out, M),
        )

    def forward(self, x):
        x = self.up(x)
        return x



class IGCV1_UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(IGCV1_UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = nn.Sequential(
            conv3x3(ch_in=1, ch_out=64),
            IGCV_block(ch_in=64, ch_out=64, M=8)
        )
        self.Conv2 = double_conv(ch_in=64, ch_out=128, M=8)
        self.Conv3 = double_conv(ch_in=128, ch_out=256, M=16)
        self.Conv4 = double_conv(ch_in=256, ch_out=512, M=16)
        self.Conv5 = double_conv(ch_in=512, ch_out=1024, M=32)

        self.Up5 = up_conv(ch_in=1024, ch_out=512, M=32)
        self.Up_conv5 = double_conv(ch_in=1024, ch_out=512, M=32)

        self.Up4 = up_conv(ch_in=512, ch_out=256, M=16)
        self.Up_conv4 = double_conv(ch_in=512, ch_out=256, M=16)

        self.Up3 = up_conv(ch_in=256, ch_out=128, M=16)
        self.Up_conv3 = double_conv(ch_in=256, ch_out=128, M=16)

        self.Up2 = up_conv(ch_in=128, ch_out=64, M=8)
        self.Up_conv2 = double_conv(ch_in=128, ch_out=64, M=8)

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