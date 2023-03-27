import torch
from torch import nn
import torch.nn.functional as F


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SCF_block(nn.Module):
    def __init__(self, ch_1, ch_2, ch_int, ch_out, drop_rate=0.):
        super(SCF_block, self).__init__()

        # channel attention for F_g, use SE Block
        # self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial_g = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.spatial_x = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)
        self.DAS = DAS_block(ch_out)
        self.attention_g = Attention(name='scse', in_channels=ch_1 + 1)
        self.attention_x = Attention(name='scse', in_channels=ch_2 + 1)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)  # b,2,h,w
        g = self.spatial_g(g)  # b,1,h,w
        # g = self.sigmoid(g) * g_in  # b,ch_1,h,w

        # channel attetion for transformer branch
        x_in = x
        # x = x.mean((2, 3), keepdim=True)  # b,ch_2,1,1
        x = self.compress(x)
        x = self.spatial_x(x)
        # x = self.spatial(x)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)  # b,ch_2,1,1
        # x = self.sigmoid(x) * x_in  # b,ch2_,h,w
        g_out = self.attention_g(torch.cat([x, g_in], dim=1))[:, 1:, :, :]
        x_out = self.attention_x(torch.cat([g, x_in], dim=1))[:, 1:, :, :]

        fuse = self.residual(torch.cat([g_out, x_out, bp], 1))

        if self.drop_rate > 0:
            out = self.dropout(fuse)
        else:
            out = fuse
        # out = self.DAS(out)
        return out


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        a = self.cSE(x)
        c = self.sSE(x)
        return x * self.cSE(x) + x * self.sSE(x)


class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())


class DAS_block(nn.Module):
    def __init__(self, channel):
        super(DAS_block, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3),
            nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3),
            nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3),
            nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5),
            nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        )
        self.asp = ASPPPooling(channel, channel)
        self.project = nn.Sequential(
            nn.Conv2d(5*channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv2(x)
        out_3 = self.conv3(x)
        out_4 = self.conv4(x)
        out = out_1 + out_2 + out_3 + out_4 + x
        return out