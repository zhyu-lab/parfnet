# -*- coding: utf-8 -*-
# @Author  : LebronMX
# @Time    : 2024/10/23 15:53
# @File    : PARFNet.py
# @Software: PyCharm



from torch import nn

from model.TCM.TCM import *

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PARFNet(nn.Module):
    def __init__(self, n_channels, n_classes, img_size, bilinear=False):
        super(PARFNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.bilinear = bilinear

        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        head_dim = [8, 16, 16, 8]
        channles = [64, 128, 256, 512, 1024 ,512, 256, 128 ,64]

        dpr = 0.1
        self.window_size = 7


        self.w_down3 =  ConvTransBlock(channles[1], channles[1], head_dim[0], self.window_size, dpr, self.img_size//4, 'W')  # 56
        self.sw_down3 = ConvTransBlock(channles[1], channles[1], head_dim[0], self.window_size, dpr, self.img_size//4, 'SW')

        self.w_down4 = ConvTransBlock(channles[2], channles[2], head_dim[1], self.window_size, dpr,  self.img_size//8, 'W')  # 28
        self.sw_down4 = ConvTransBlock(channles[2], channles[2], head_dim[1], self.window_size, dpr, self.img_size//8, 'SW')

        self.w_up1 =  ConvTransBlock(channles[2], channles[2], head_dim[2], self.window_size, dpr, self.img_size//8, 'W')
        self.sw_up1 = ConvTransBlock(channles[2], channles[2], head_dim[2], self.window_size, dpr, self.img_size//8, 'SW')

        self.w_up2 = ConvTransBlock(channles[1], channles[1], head_dim[3], self.window_size, dpr,  self.img_size//4, 'W')
        self.sw_up2 = ConvTransBlock(channles[1], channles[1], head_dim[3], self.window_size, dpr, self.img_size//4, 'SW')




        self.inc = (DoubleConv(n_channels, 64))
        self.dc1 = DynamicConvolution(channles[0], channles[0])

        self.down1 = (Down(channles[0], channles[1]))
        self.dc2 = DynamicConvolution(channles[1], channles[1])

        self.down2 = (Down(channles[1], channles[2]))
        self.down3 = (Down(channles[2], channles[3]))
        factor = 2 if bilinear else 1
        self.down4 = (Down(channles[3], channles[4] // factor))


        self.up1 = (Up(channles[4], channles[5] // factor, bilinear))
        self.up2 = (Up(channles[5], channles[6] // factor, bilinear))
        self.up3 = (Up(channles[6], channles[7] // factor, bilinear))
        self.up4 = (Up(channles[7], channles[8], bilinear))

        self.outc = (OutConv(channles[8], n_classes))
        self.last_activation = nn.Sigmoid

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.dc1(x1) + x1

        x2 = self.down1(x1)
        x2 = self.dc2(x2) + x2

        x3 = self.down2(x2)
        x3s = self.w_down3(x3)
        x3 = self.sw_down3(x3s)


        x4 = self.down3(x3)
        x4s = self.w_down4(x4)
        x4 = self.sw_down4(x4s)

        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x1us = self.w_up1(x)
        x = self.sw_up1(x1us)

        x = self.up2(x, x3)
        x2us = self.w_up2(x)
        x = self.sw_up2(x2us)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits


