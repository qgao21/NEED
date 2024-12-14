import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.inc = nn.Sequential(
            single_conv(in_channels, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.mlp1 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 64)
        )
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.mlp2 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.mlp3 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.mlp4 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 64)
        )
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, out_channels)

    def forward(self, x, t):
        inx = self.inc(x)
        time_emb = self.time_mlp(t)
        down1 = self.down1(inx)

        condition1 = self.mlp1(time_emb)
        b, c = condition1.shape
        condition1 = rearrange(condition1, 'b c -> b c 1 1')

        down1 = down1 + condition1
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        condition2 = self.mlp2(time_emb)
        b, c = condition2.shape
        condition2 = rearrange(condition2, 'b c -> b c 1 1')

        down2 = down2 + condition2
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        condition3 = self.mlp3(time_emb)
        b, c = condition3.shape
        condition3 = rearrange(condition3, 'b c -> b c 1 1')

        up1 = up1 + condition3
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        condition4 = self.mlp4(time_emb)
        b, c = condition4.shape
        condition4 = rearrange(condition4, 'b c -> b c 1 1')

        up2 = up2 + condition4
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out


class Network(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, context=True):
        super(Network, self).__init__()
        self.context = context
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, t):
        if self.context:
            x_middle = x[:, 1].unsqueeze(1)
        else:
            x_middle = x

        out = self.unet(x, t) + x_middle

        return out