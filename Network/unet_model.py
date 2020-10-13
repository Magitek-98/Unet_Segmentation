# full assembly of the sub-parts to form the complete net
# sub-parts of the U-Net model

import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]


        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.inc = inconv(1, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        logging.debug((x.shape, "UNet input"))
        x1 = self.inc(x)
        logging.debug((x1.shape, "UNet x1 (input->inconv)"))
        x2 = self.down1(x1)
        logging.debug((x2.shape, "UNet x2 (x1->down1)"))
        x3 = self.down2(x2)
        logging.debug((x3.shape, "UNet x3 (x2->down2)"))
        x4 = self.down3(x3)
        logging.debug((x4.shape, "UNet x4 (x3->down3)"))
        x5 = self.down4(x4)
        logging.debug((x5.shape, "UNet x5 (x4->down4)"))
        x = self.up1(x5, x4)
        logging.debug((x.shape, "UNet up1 (x5,x4->up1)"))
        x = self.up2(x, x3)
        logging.debug((x.shape, "UNet up2 (up1,x3->up2)"))
        x = self.up3(x, x2)
        logging.debug((x.shape, "UNet up3 (up2,x2->up3)"))
        x = self.up4(x, x1)
        logging.debug((x.shape, "UNet up4 (up3,x1->up4"))
        x = self.outc(x)
        logging.debug((x.shape, "UNet output"))
        return x


if __name__ == '__main__':
    # DEBUG
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = UNet()
    net(torch.randn(1, 1, 192, 192))

