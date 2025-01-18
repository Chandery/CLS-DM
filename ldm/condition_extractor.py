# 2D-Unet Model taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UnetEncoder(nn.Module):

    def __init__(self, in_channels):
        super(UnetEncoder, self).__init__()
        self.n_channels = in_channels

        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 128)
        self.down4 = Down(128, 64)
        self.out_conv = nn.Conv2d(in_channels=64,out_channels=8,kernel_size=1,stride=1)
        # self.up1 = Up(1024, 256)
        # self.up2 = Up(512, 128)
        # self.up3 = Up(256, 64)
        # self.up4 = Up(128, 64)
        # self.is_seg = is_seg
        # if self.is_seg:
        #     self.seg_outc = OutConv(64, classes)
        # else:
        #     self.outc = OutConv(64, classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.out_conv(x)
        _,_,h,w =  x.shape
        x = x.unsqueeze(-1)
        x = x.repeat(1,1,1,1,h)

        return x


if __name__ == '__main__':
    model = UnetEncoder(in_channels=3)
    test_tensor = torch.randn((1,3,256,256))
    out = model(test_tensor)
    print(out.shape)