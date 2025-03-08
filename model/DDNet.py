#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright:    WZP
Filename:     DDNet.py
Description:  Deformation Detection Network (DDNet) for large-scale interferograms.
"""

__all__ = ["DDNet"]

import torch
import torch.nn as nn
from torchsummary import summary


class DilatedBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DilatedBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 3), padding=(2, 2), dilation=(2, 2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 3), padding=(3, 3), dilation=(3, 3)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Conv2d(out_ch * 3, out_ch, (3, 3), padding=(1, 1), dilation=(1, 1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        o1 = self.layer1(x)
        o2 = self.layer2(x)
        o3 = self.layer3(x)
        o = torch.cat((o1, o2, o3), dim=1)
        o = self.relu(self.bn(x + self.conv(o)))
        return o


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(out_ch, out_ch, (3, 3), padding=(1, 1), dilation=(1, 1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(x + self.conv(x)))


class DDNet(nn.Module):
    def __init__(self, num_channels):
        super(DDNet, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        # Use 4 dilated blocks instead of 8
        self.dilated = nn.Sequential(
            *[DilatedBlock(64, 64) for _ in range(4)]
        )

        # Use 4 residual blocks instead of 10
        self.residual = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(4)]
        )

        self.outc = nn.Conv2d(64, num_channels, (3, 3), padding=(1, 1))

    def forward(self, x):
        x = self.inc(x)
        x = self.dilated(x)
        x = self.residual(x)
        x = self.outc(x)
        return x


"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDNet(num_channels=1).to(device)
    summary(model, (1, 180, 180))  # Input size can be adjusted for full-sized interferograms