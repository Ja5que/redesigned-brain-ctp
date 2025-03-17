# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import math
import torch.autograd as autograd
import torch.nn.functional as F


# input (Tensor)
# pad (tuple)
# mode – 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
# value – fill value for 'constant' padding. Default: 0

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, dilation=1, stride=1, groups=1, chls_group=12, is_bn=True,
                 is_relu=True):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride,
                              dilation=dilation, groups=groups, bias=True)
        self.bn = None
        self.relu = None

        if is_bn is True:
            self.bn = nn.BatchNorm2d(out_chl)
            # self.bn = nn.InstanceNorm2d(out_chl)
            # self.bn = nn.GroupNorm(num_groups=out_chl//chls_group, num_channels=out_chl)
        if is_relu is True:
            self.relu = nn.LeakyReLU(inplace=True)
            # self.relu = nn.GELU()
            # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackEncoder, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

    def forward(self, x):
        conv_out = self.encode(x)
        down_out = F.max_pool2d(conv_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return conv_out, down_out


class StackDecoder(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackDecoder, self).__init__()
        self.conv = nn.Sequential(
            ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        )

    def forward(self, up_in, conv_res):
        _, _, H, W = conv_res.size()
        up_out = F.upsample(up_in, size=(H, W), mode='bilinear')
        conv_out = self.conv(torch.cat([up_out, conv_res], 1))
        return conv_out


class StackResEncoder2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackResEncoder2d, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False),
        )

        self.convx = None

        if in_chl != out_chl:
            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                      is_relu=False)

    def forward(self, x):

        if self.convx is None:
            conv_out = F.leaky_relu(self.encode(x) + x)
        else:
            conv_out = F.leaky_relu(self.encode(x) + self.convx(x))

        down_out = F.max_pool2d(conv_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return conv_out, down_out


class StackResCenter2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackResCenter2d, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False),
        )

        self.convx = None

        if in_chl != out_chl:
            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                      is_relu=False)

    def forward(self, x):

        if self.convx is None:
            conv_out = F.leaky_relu(self.encode(x) + x)
        else:
            conv_out = F.leaky_relu(self.encode(x) + self.convx(x))

        return conv_out


class StackResDecoder2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackResDecoder2d, self).__init__()

        self.conv1 = ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv2 = ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1,
                                  is_relu=False)
        self.convx = ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                  is_relu=False)

    def forward(self, up_in, conv_res):
        _, _, H, W = conv_res.size()
        up_out = F.upsample(up_in, size=(H, W), mode='bilinear')
        conv1 = self.conv1(torch.cat([up_out, conv_res], 1))
        conv2 = self.conv2(conv1)
        convx = F.leaky_relu(conv2 + self.convx(torch.cat([up_out, conv_res], 1)))
        return convx


class UNet2d(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=60):
        super(UNet2d, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackEncoder(self.model_chl, self.model_chl, kernel_size=3)  # 256
        self.down2 = StackEncoder(self.model_chl * 1, self.model_chl * 2, kernel_size=3)  # 128
        self.down3 = StackEncoder(self.model_chl * 2, self.model_chl * 4, kernel_size=3)  # 64
        self.down4 = StackEncoder(self.model_chl * 4, self.model_chl * 8, kernel_size=3)  # 32

        self.center = nn.Sequential(ConvBnRelu2d(self.model_chl * 8, self.model_chl * 16, kernel_size=3, stride=1),
                                    ConvBnRelu2d(self.model_chl * 16, self.model_chl * 16, kernel_size=3, stride=1))

        self.up4 = StackDecoder(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackDecoder(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackDecoder(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackDecoder(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(
            ConvBnRelu2d(self.model_chl, self.out_chl, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)
        conv2, d2 = self.down2(d1)
        conv3, d3 = self.down3(d2)
        conv4, d4 = self.down4(d3)
        conv5 = self.center(d4)
        up4 = self.up4(conv5, conv4)
        up3 = self.up3(up4, conv3)
        up2 = self.up2(up3, conv2)
        up1 = self.up1(up2, conv1)
        conv6 = self.end(up1)
        res_out = F.leaky_relu(x + conv6)
        return res_out


class UNet2d(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=60):
        super(UNet2d, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackEncoder(self.model_chl, self.model_chl, kernel_size=3)  # 256
        self.down2 = StackEncoder(self.model_chl * 1, self.model_chl * 2, kernel_size=3)  # 128
        self.down3 = StackEncoder(self.model_chl * 2, self.model_chl * 4, kernel_size=3)  # 64
        self.down4 = StackEncoder(self.model_chl * 4, self.model_chl * 8, kernel_size=3)  # 32

        self.center = nn.Sequential(ConvBnRelu2d(self.model_chl * 8, self.model_chl * 16, kernel_size=3, stride=1),
                                    ConvBnRelu2d(self.model_chl * 16, self.model_chl * 16, kernel_size=3, stride=1))

        self.up4 = StackDecoder(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackDecoder(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackDecoder(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackDecoder(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(
            ConvBnRelu2d(self.model_chl, self.out_chl, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)
        conv2, d2 = self.down2(d1)
        conv3, d3 = self.down3(d2)
        conv4, d4 = self.down4(d3)
        conv5 = self.center(d4)
        up4 = self.up4(conv5, conv4)
        up3 = self.up3(up4, conv3)
        up2 = self.up2(up3, conv2)
        up1 = self.up1(up2, conv1)
        conv6 = self.end(up1)
        res_out = F.leaky_relu(x + conv6)
        return res_out


class Registration(nn.Module):
    def __init__(self, C, mode='bilinear'):
        super(Registration, self).__init__()
        self.mode = mode
        # Learnable rotation and shift parameters for each channel only
        self.rotate = nn.Parameter(torch.zeros(C) + 0.1 * math.pi / 180, requires_grad=True)
        self.shift_x = nn.Parameter(torch.zeros(C) + 0.01, requires_grad=True)
        self.shift_y = nn.Parameter(torch.zeros(C) + 0.01, requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # Calculate rotation matrix for each channel (C)
        cos_theta = torch.cos(self.rotate).unsqueeze(0).unsqueeze(-1)
        sin_theta = torch.sin(self.rotate).unsqueeze(0).unsqueeze(-1)

        # Prepare the matrix with size (C, 2, 3) to broadcast over the batch dimension
        matrix = torch.zeros(C, 2, 3, device=x.device, dtype=x.dtype)
        matrix[:, 0, 0] = cos_theta.squeeze()
        matrix[:, 0, 1] = -sin_theta.squeeze()
        matrix[:, 0, 2] = self.shift_x
        matrix[:, 1, 0] = sin_theta.squeeze()
        matrix[:, 1, 1] = cos_theta.squeeze()
        matrix[:, 1, 2] = self.shift_y

        # Reshape input and matrix for grid_sample
        x = x.view(B * C, 1, H, W)
        matrix = matrix.repeat(B, 1, 1)  # Repeat matrix B times to apply the same transformation across the batch

        # Generate affine grid for each (B * C)
        affine_grid = F.affine_grid(matrix, x.size(), align_corners=True)

        # Apply the transformation for each channel independently
        x_transformed = F.grid_sample(x, affine_grid, mode=self.mode, align_corners=True)

        # Reshape back to (B, C, H, W)
        x_transformed = x_transformed.view(B, C, H, W)

        return x_transformed


class UNet2dRegis(nn.Module):
    def __init__(self, in_chl, out_chl, model_chl):
        super(UNet2dRegis, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.regis = Registration(30)

        self.begin = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackEncoder(self.model_chl, self.model_chl, kernel_size=3)  # 256
        self.down2 = StackEncoder(self.model_chl * 1, self.model_chl * 2, kernel_size=3)  # 128
        self.down3 = StackEncoder(self.model_chl * 2, self.model_chl * 4, kernel_size=3)  # 64
        self.down4 = StackEncoder(self.model_chl * 4, self.model_chl * 8, kernel_size=3)  # 32

        self.center = nn.Sequential(ConvBnRelu2d(self.model_chl * 8, self.model_chl * 16, kernel_size=3, stride=1),
                                    ConvBnRelu2d(self.model_chl * 16, self.model_chl * 16, kernel_size=3, stride=1))

        self.up4 = StackDecoder(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackDecoder(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackDecoder(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackDecoder(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(
            ConvBnRelu2d(self.model_chl, self.out_chl, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)
        conv2, d2 = self.down2(d1)
        conv3, d3 = self.down3(d2)
        conv4, d4 = self.down4(d3)
        conv5 = self.center(d4)
        up4 = self.up4(conv5, conv4)
        up3 = self.up3(up4, conv3)
        up2 = self.up2(up3, conv2)
        up1 = self.up1(up2, conv1)
        conv6 = self.end(up1)
        res_out = F.leaky_relu(x + conv6)

        res_out = self.regis(res_out)

        return res_out



class ResUNet2d(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super(ResUNet2d, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackResEncoder2d(self.model_chl, self.model_chl, kernel_size=3)  # 256
        self.down2 = StackResEncoder2d(self.model_chl * 1, self.model_chl * 2, kernel_size=3)  # 128
        self.down3 = StackResEncoder2d(self.model_chl * 2, self.model_chl * 4, kernel_size=3)  # 64
        self.down4 = StackResEncoder2d(self.model_chl * 4, self.model_chl * 8, kernel_size=3)  # 32

        self.center = StackResCenter2d(self.model_chl * 8, self.model_chl * 16, kernel_size=3)  # 32

        self.up4 = StackResDecoder2d(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackResDecoder2d(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackResDecoder2d(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackResDecoder2d(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(
            ConvBnRelu2d(self.model_chl, self.out_chl, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)
        conv2, d2 = self.down2(d1)
        conv3, d3 = self.down3(d2)
        conv4, d4 = self.down4(d3)
        conv5 = self.center(d4)
        up4 = self.up4(conv5, conv4)
        up3 = self.up3(up4, conv3)
        up2 = self.up2(up3, conv2)
        up1 = self.up1(up2, conv1)
        conv6 = self.end(up1)
        res_out = F.leaky_relu(x + conv6)
        return res_out


class UNet2dDeepSuperviser(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super(UNet2dDeepSuperviser, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackEncoder(self.model_chl, self.model_chl, kernel_size=3)  # 256
        self.down2 = StackEncoder(self.model_chl * 1, self.model_chl * 2, kernel_size=3)  # 128
        self.down3 = StackEncoder(self.model_chl * 2, self.model_chl * 4, kernel_size=3)  # 64
        self.down4 = StackEncoder(self.model_chl * 4, self.model_chl * 8, kernel_size=3)  # 32

        self.center = nn.Sequential(ConvBnRelu2d(self.model_chl * 8, self.model_chl * 16, kernel_size=3, stride=1),
                                    ConvBnRelu2d(self.model_chl * 16, self.model_chl * 16, kernel_size=3, stride=1))

        self.up4 = StackDecoder(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackDecoder(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackDecoder(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackDecoder(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(ConvBnRelu2d(self.model_chl, 1, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)
        conv2, d2 = self.down2(d1)
        conv3, d3 = self.down3(d2)
        conv4, d4 = self.down4(d3)
        conv5 = self.center(d4)
        up4 = self.up4(conv5, conv4)
        up3 = self.up3(up4, conv3)
        up2 = self.up2(up3, conv2)
        up1 = self.up1(up2, conv1)
        conv6 = self.end(up1)
        res_out = F.leaky_relu(x + conv6)
        return conv5, up4, up3, up2, res_out


class StackDenseEncoder2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackDenseEncoder2d, self).__init__()

        self.conv1 = ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv2 = ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv3 = ConvBnRelu2d(in_chl + out_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1,
                                  groups=1, is_relu=False)

        self.convx = None

        if in_chl != out_chl:
            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                      is_relu=False)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat([x, conv1], 1))
        conv3 = self.conv3(torch.cat([x, conv1, conv2], 1))

        if self.convx is None:
            convx = F.leaky_relu(conv3 + x)
        else:
            convx = F.leaky_relu(conv3 + self.convx(x))

        down_out = F.max_pool2d(convx, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return convx, down_out


class StackDenseBlock2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackDenseBlock2d, self).__init__()

        self.conv1 = ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv2 = ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv3 = ConvBnRelu2d(in_chl + out_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1,
                                  groups=1, is_relu=False)

        self.convx = None

        if in_chl != out_chl:
            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                      is_relu=False)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat([x, conv1], 1))
        conv3 = self.conv3(torch.cat([x, conv1, conv2], 1))

        if self.convx is None:
            convx = F.leaky_relu(conv3 + x)
        else:
            convx = F.leaky_relu(conv3 + self.convx(x))

        return convx


class AIO(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=64, degration_type_num=3):
        super(AIO, self).__init__()

        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl
        self.degradation_type_num = degration_type_num

        self.classfier = Degradation_Classfier(self.in_chl, self.degradation_type_num, self.model_chl)  # 退化分类器
        self.begin = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackDenseEncoder2d(self.model_chl, self.model_chl, kernel_size=3)
        self.down2 = StackDenseEncoder2d(self.model_chl * 1 + self.model_chl * 1, self.model_chl * 2, kernel_size=3)
        self.down3 = StackDenseEncoder2d(self.model_chl * 2 + self.model_chl * 2, self.model_chl * 4, kernel_size=3)
        self.down4 = StackDenseEncoder2d(self.model_chl * 4 + self.model_chl * 4, self.model_chl * 8, kernel_size=3)

        self.center = StackDenseBlock2d(self.model_chl * 8 + self.model_chl * 8, self.model_chl * 16, kernel_size=3)

        self.up4 = StackResDecoder2d(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackResDecoder2d(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackResDecoder2d(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackResDecoder2d(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(
            ConvBnRelu2d(self.model_chl, self.out_chl, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        # model_chl;model_chl*2;model_chl*4;model_chl*8
        feature1, feature2, feature3, feature4, predictType = self.classfier(x)

        conv0 = self.begin(x)

        # print(conv0.shape)
        conv1, d1 = self.down1(conv0)

        # print(d1.shape)
        # print(feature1.shape)
        conv2, d2 = self.down2(torch.cat((d1, feature1), dim=1))

        # print(d2.shape)
        # print(feature2.shape)
        conv3, d3 = self.down3(torch.cat((d2, feature2), dim=1))

        # print(d3.shape)
        # print(feature3.shape)
        conv4, d4 = self.down4(torch.cat((d3, feature3), dim=1))

        # print(d4.shape)
        # print(feature4.shape)
        conv5 = self.center(torch.cat((d4, feature4), dim=1))

        up4 = self.up4(conv5, conv4)
        up3 = self.up3(up4, conv3)
        up2 = self.up2(up3, conv2)
        up1 = self.up1(up2, conv1)
        conv6 = self.end(up1)
        res_out = F.leaky_relu(x + conv6)

        # conv1, d1 = self.down1(conv0)
        # conv2, d2 = self.down2(d1)
        # conv3, d3 = self.down3(d2)
        # conv4, d4 = self.down4(d3)
        # conv5 = self.center(d4)
        # up4 = self.up4(conv5, conv4)
        # up3 = self.up3(up4, conv3)
        # up2 = self.up2(up3, conv2)
        # up1 = self.up1(up2, conv1)
        # conv6 = self.end(up1)
        # res_out = F.leaky_relu(x+conv6)
        return res_out, predictType


# 基于Discriminator修改的Degradation_Classfier，有BUG没改。
class Degradation_Classfier(nn.Module):
    def __init__(self, in_chl=1, out_chl=3, model_chl=32):
        super(Degradation_Classfier, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl
        self.ConvLayers1 = nn.Sequential(
            ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1),
            ConvBnRelu2d(self.model_chl, self.model_chl, kernel_size=3, stride=2, is_bn=False, is_relu=False))

        self.ConvLayers2 = nn.Sequential(
            ConvBnRelu2d(self.model_chl, self.model_chl * 2, kernel_size=3, stride=1),
            ConvBnRelu2d(self.model_chl * 2, self.model_chl * 2, kernel_size=3, stride=2, is_bn=False, is_relu=False))

        self.ConvLayers3 = nn.Sequential(
            ConvBnRelu2d(self.model_chl * 2, self.model_chl * 4, kernel_size=3, stride=1),
            ConvBnRelu2d(self.model_chl * 4, self.model_chl * 4, kernel_size=3, stride=2, is_bn=False, is_relu=False))

        self.ConvLayers4 = nn.Sequential(
            ConvBnRelu2d(self.model_chl * 4, self.model_chl * 8, kernel_size=3, stride=1),
            ConvBnRelu2d(self.model_chl * 8, self.model_chl * 8, kernel_size=3, stride=2, is_bn=False, is_relu=False))

        self.FCLayer = nn.Sequential(
            nn.Linear(self.model_chl * 8, self.out_chl)
        )

    def forward(self, x):
        feature1 = self.ConvLayers1(x)
        feature2 = self.ConvLayers2(feature1)
        feature3 = self.ConvLayers3(feature2)
        feature4 = self.ConvLayers4(feature3)

        out = torch.reshape(F.adaptive_avg_pool2d(feature4, (1, 1)), [feature4.shape[0], feature4.shape[1]])
        out = self.FCLayer(out)
        out = nn.functional.softmax(out, dim=1)
        # print(out.sum(1))  #判断每张图像的概率分布是否满足和为1的条件
        return feature1, feature2, feature3, feature4, out


class DenseUNet2d(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=64):
        super(DenseUNet2d, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackDenseEncoder2d(self.model_chl, self.model_chl, kernel_size=3)
        self.down2 = StackDenseEncoder2d(self.model_chl * 1, self.model_chl * 2, kernel_size=3)
        self.down3 = StackDenseEncoder2d(self.model_chl * 2, self.model_chl * 4, kernel_size=3)
        self.down4 = StackDenseEncoder2d(self.model_chl * 4, self.model_chl * 8, kernel_size=3)

        self.center = StackDenseBlock2d(self.model_chl * 8, self.model_chl * 16, kernel_size=3)

        self.up4 = StackResDecoder2d(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackResDecoder2d(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackResDecoder2d(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackResDecoder2d(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(
            ConvBnRelu2d(self.model_chl, self.out_chl, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)
        conv2, d2 = self.down2(d1)
        conv3, d3 = self.down3(d2)
        conv4, d4 = self.down4(d3)
        conv5 = self.center(d4)
        up4 = self.up4(conv5, conv4)
        up3 = self.up3(up4, conv3)
        up2 = self.up2(up3, conv2)
        up1 = self.up1(up2, conv1)
        conv6 = self.end(up1)
        res_out = F.leaky_relu(x + conv6)
        return res_out


class Discriminator2D(nn.Module):
    def __init__(self, in_chl=5, out_chl=1, model_chl=32):
        super(Discriminator2D, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl
        self.ConvLayers = nn.Sequential(
            ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1),
            ConvBnRelu2d(self.model_chl, self.model_chl, kernel_size=3, stride=2, is_bn=False, is_relu=False),

            ConvBnRelu2d(self.model_chl, self.model_chl * 2, kernel_size=3, stride=1),
            ConvBnRelu2d(self.model_chl * 2, self.model_chl * 2, kernel_size=3, stride=2, is_bn=False, is_relu=False),

            ConvBnRelu2d(self.model_chl * 2, self.model_chl * 4, kernel_size=3, stride=1),
            ConvBnRelu2d(self.model_chl * 4, self.model_chl * 4, kernel_size=3, stride=2, is_bn=False, is_relu=False),

            ConvBnRelu2d(self.model_chl * 4, self.model_chl * 8, kernel_size=3, stride=1),
            ConvBnRelu2d(self.model_chl * 8, self.model_chl * 8, kernel_size=3, stride=2, is_bn=False, is_relu=False),
        )
        self.FCLayer = nn.Sequential(
            nn.Linear(self.model_chl * 8, self.out_chl)
        )

    def forward(self, x):
        out = self.ConvLayers(x)
        out = torch.reshape(F.adaptive_avg_pool2d(out, (1, 1)), [out.shape[0], out.shape[1]])
        out = self.FCLayer(out)

        return out


class DiscriminatorCLRT(nn.Module):
    def __init__(self, in_chl=5, out_chl=1, model_chl=32):
        super(DiscriminatorCLRT, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl
        self.conv1 = ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1)
        self.conv2 = ConvBnRelu2d(self.model_chl, self.model_chl, kernel_size=3, stride=2, is_bn=False, is_relu=False)
        self.conv3 = ConvBnRelu2d(self.model_chl, self.model_chl * 2, kernel_size=3, stride=1)
        self.conv4 = ConvBnRelu2d(self.model_chl * 2, self.model_chl * 2, kernel_size=3, stride=2, is_bn=False,
                                  is_relu=False)
        self.conv5 = ConvBnRelu2d(self.model_chl * 2, self.model_chl * 4, kernel_size=3, stride=1)
        self.conv6 = ConvBnRelu2d(self.model_chl * 4, self.model_chl * 4, kernel_size=3, stride=2, is_bn=False,
                                  is_relu=False)
        self.conv7 = ConvBnRelu2d(self.model_chl * 4, self.model_chl * 8, kernel_size=3, stride=1)
        self.conv8 = ConvBnRelu2d(self.model_chl * 8, self.model_chl * 8, kernel_size=3, stride=2, is_bn=False,
                                  is_relu=False)
        # self.conv9 = ConvBnRelu2d(self.model_chl * 8, self.model_chl * 8, kernel_size=3, stride=1)
        self.FCLayer = nn.Sequential(nn.Linear(self.model_chl * 8, self.out_chl))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        # x9 = self.conv9(x8)
        out = torch.reshape(F.adaptive_avg_pool2d(x8, (1, 1)), [x8.shape[0], x8.shape[1]])
        out = self.FCLayer(out)

        return out, x2, x4, x6, x8


class DiscriminatorCL(nn.Module):
    def __init__(self, in_chl=5, out_chl=1, model_chl=40, chls_group=8):
        super(DiscriminatorCL, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl
        self.chls_group = chls_group

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_chl, self.model_chl, kernel_size=3, padding=(3 - 1) // 2, stride=1, dilation=1, groups=1,
                      bias=False),
            nn.BatchNorm2d(self.model_chl),
            nn.LeakyReLU(),
            nn.Conv2d(self.model_chl, self.model_chl, kernel_size=3, padding=(3 - 1) // 2, stride=2, dilation=1,
                      groups=1, bias=True),

            nn.Conv2d(self.model_chl * 1, self.model_chl * 2, kernel_size=3, padding=(3 - 1) // 2, stride=1, dilation=1,
                      groups=1, bias=False),
            nn.BatchNorm2d(self.model_chl * 2),
            nn.LeakyReLU(),
            nn.Conv2d(self.model_chl * 2, self.model_chl * 2, kernel_size=3, padding=(3 - 1) // 2, stride=2, dilation=1,
                      groups=1, bias=True),

            nn.Conv2d(self.model_chl * 2, self.model_chl * 4, kernel_size=3, padding=(3 - 1) // 2, stride=1, dilation=1,
                      groups=1, bias=False),
            nn.BatchNorm2d(self.model_chl * 4),
            nn.LeakyReLU(),
            nn.Conv2d(self.model_chl * 4, self.model_chl * 4, kernel_size=3, padding=(3 - 1) // 2, stride=2, dilation=1,
                      groups=1, bias=True),

            nn.Conv2d(self.model_chl * 4, self.model_chl * 8, kernel_size=3, padding=(3 - 1) // 2, stride=1, dilation=1,
                      groups=1, bias=False),
            nn.BatchNorm2d(self.model_chl * 8),
            nn.LeakyReLU(),
            nn.Conv2d(self.model_chl * 8, self.model_chl * 8, kernel_size=3, padding=(3 - 1) // 2, stride=2, dilation=1,
                      groups=1, bias=True),
            )
        self.FCLayer = nn.Sequential(
            nn.Linear(self.model_chl * 8, self.out_chl),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = torch.reshape(F.adaptive_avg_pool2d(out, (1, 1)), [out.shape[0], out.shape[1]])
        out = self.FCLayer(out)
        return out


# class DiscriminatorV2(nn.Module):
#     def __init__(self, in_chl=5, out_chl=1, model_chl=40, chls_group=8):
#         super(DiscriminatorV2, self).__init__()
#         self.out_chl = out_chl
#         self.model_chl = model_chl
#         self.in_chl = in_chl
#         self.chls_group = chls_group

#         self.encoder = nn.Sequential(nn.Conv2d(self.in_chl, self.model_chl, kernel_size=3, padding=(3-1)//2, stride=1, dilation=1, groups=1, bias=True),
#                                      nn.GroupNorm(num_groups=self.model_chl * 1 // self.chls_group, num_channels=self.model_chl * 1),
#                                      nn.ReLU(),
#                                      nn.Conv2d(self.model_chl, self.model_chl, kernel_size=3, padding=(3-1)//2, stride=1, dilation=1, groups=1, bias=True),
#                                      nn.GroupNorm(num_groups=self.model_chl * 1 // self.chls_group, num_channels=self.model_chl * 1),
#                                      nn.ReLU(),
#                                      nn.MaxPool2d(kernel_size=2, stride=2),

#                                      nn.Conv2d(self.model_chl*1, self.model_chl*2, kernel_size=3, padding=(3-1)//2, stride=1, dilation=1, groups=1, bias=True),
#                                      nn.GroupNorm(num_groups=self.model_chl * 2 // self.chls_group, num_channels=self.model_chl * 2),
#                                      nn.ReLU(),
#                                      nn.Conv2d(self.model_chl*2, self.model_chl*2, kernel_size=3, padding=(3-1)//2, stride=1, dilation=1, groups=1, bias=True),
#                                      nn.GroupNorm(num_groups=self.model_chl * 2 // self.chls_group, num_channels=self.model_chl * 2),
#                                      nn.ReLU(),
#                                      nn.MaxPool2d(kernel_size=2, stride=2),

#                                      nn.Conv2d(self.model_chl*2, self.model_chl*4, kernel_size=3, padding=(3-1)//2, stride=1, dilation=1, groups=1, bias=True),
#                                      nn.GroupNorm(num_groups=self.model_chl * 4 // self.chls_group, num_channels=self.model_chl * 4),
#                                      nn.ReLU(),
#                                      nn.Conv2d(self.model_chl*4, self.model_chl*4, kernel_size=3, padding=(3-1)//2, stride=1, dilation=1, groups=1, bias=True),
#                                      nn.GroupNorm(num_groups=self.model_chl * 4 // self.chls_group, num_channels=self.model_chl * 4),
#                                      nn.ReLU(),
#                                      nn.MaxPool2d(kernel_size=2, stride=2),

#                                      nn.Conv2d(self.model_chl*4, self.model_chl*8, kernel_size=3, padding=(3-1)//2, stride=1, dilation=1, groups=1, bias=True),
#                                      nn.GroupNorm(num_groups=self.model_chl * 8 // self.chls_group, num_channels=self.model_chl * 8),
#                                      nn.ReLU(),
#                                      nn.Conv2d(self.model_chl*8, self.model_chl*8, kernel_size=3, padding=(3-1)//2, stride=1, dilation=1, groups=1, bias=True),
#                                      nn.GroupNorm(num_groups=self.model_chl * 8 // self.chls_group, num_channels=self.model_chl * 8),
#                                      nn.ReLU(),
#                                      nn.MaxPool2d(kernel_size=2, stride=2),

#                                      nn.Conv2d(self.model_chl*8, self.model_chl*16, kernel_size=3, padding=(3-1)//2, stride=1, dilation=1, groups=1, bias=True),
#                                      nn.GroupNorm(num_groups=self.model_chl * 16 // self.chls_group, num_channels=self.model_chl * 16),
#                                      nn.ReLU(),
#                                      nn.Conv2d(self.model_chl*16, self.model_chl*16, kernel_size=3, padding=(3-1)//2, stride=1, dilation=1, groups=1, bias=True),
#                                      nn.GroupNorm(num_groups=self.model_chl * 16 // self.chls_group, num_channels=self.model_chl * 16),
#                                      nn.ReLU()
#                                      )
#         self.FCLayer = nn.Sequential(
#             nn.Linear(self.model_chl * 16, self.model_chl * 2),
#             nn.ReLU(),
#             nn.Linear(self.model_chl * 2, self.out_chl),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         out = self.encoder(x)
#         out = torch.reshape(F.adaptive_avg_pool2d(out, (1, 1)), [out.shape[0], out.shape[1]])
#         out = self.FCLayer(out)
#         return out

def compute_gradient_penalty(D, real_samples, fake_samples):
    Tensor = torch.cuda.FloatTensor
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    if real_samples.ndim == 5:
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1, 1))).to(
            'cuda:' + str(fake_samples.get_device()))
    else:
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to('cuda:' + str(fake_samples.get_device()))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(
        'cuda:' + str(fake_samples.get_device()))
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def compute_gradient_penalty_cl(D, real_samples, fake_samples):
    Tensor = torch.cuda.FloatTensor
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    if real_samples.ndim == 5:
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1, 1))).to(
            'cuda:' + str(fake_samples.get_device()))
    else:
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to('cuda:' + str(fake_samples.get_device()))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _, _, _, _ = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(
        'cuda:' + str(fake_samples.get_device()))
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class StackAutoEncoder(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackAutoEncoder, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

    def forward(self, x):
        conv_out = self.encode(x)
        down_out = F.max_pool2d(conv_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return down_out


class StackAutoDecoder(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackAutoDecoder, self).__init__()
        self.conv = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        )

    def forward(self, up_in, hw_size):
        up_out = F.upsample(up_in, size=hw_size, mode='bilinear')
        conv_out = self.conv(up_out)
        return conv_out


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()

        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2

        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')

        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        # ----------xy----------#
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # ----------zx----------#
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(
            F.interpolate(params_zx, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        # ----------zy----------#
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(
            F.interpolate(params_zy, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        # ----------dw----------#
        x4 = self.dw(x4)
        # ----------concat----------#
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # ----------ldw----------#
        x = self.norm2(x)
        x = self.ldw(x)
        return x


class UNetGPHAV22d(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super(UNetGPHAV22d, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackEncoder(self.model_chl, self.model_chl, kernel_size=3)  # 256
        self.down2 = StackEncoder(self.model_chl * 1, self.model_chl * 2, kernel_size=3)  # 128
        self.down3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(self.model_chl * 2, self.model_chl * 4))
        self.down4 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(self.model_chl * 4, self.model_chl * 8))
        self.center = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(self.model_chl * 8, self.model_chl * 16))

        self.up4 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(self.model_chl * (16 + 8), self.model_chl * 8))
        self.up3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(self.model_chl * (8 + 4), self.model_chl * 4))
        self.up2 = StackDecoder(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackDecoder(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(
            ConvBnRelu2d(self.model_chl, self.out_chl, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)
        conv2, d2 = self.down2(d1)
        conv3 = self.down3(d2)
        d3 = F.max_pool2d(conv3, kernel_size=2, stride=2, padding=0, ceil_mode=True)
        conv4 = self.down4(d3)
        d4 = F.max_pool2d(conv4, kernel_size=2, stride=2, padding=0, ceil_mode=True)
        conv5 = self.center(d4)
        up4 = F.upsample(conv5, size=(x.size(2) // 8, x.size(3) // 8), mode='bilinear')
        conv6 = self.up4(torch.cat([up4, conv4], 1))
        up3 = F.upsample(conv6, size=(x.size(2) // 4, x.size(3) // 4), mode='bilinear')
        conv7 = self.up3(torch.cat([up3, conv3], 1))
        up2 = self.up2(conv7, conv2)
        up1 = self.up1(up2, conv1)
        conv8 = self.end(up1)
        res_out = F.leaky_relu(x + conv8)
        return res_out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class StackResEncoderSCA2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackResEncoderSCA2d, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False),
        )

        self.csa = CBAM(out_chl)

        self.convx = None

        if in_chl != out_chl:
            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                      is_relu=False)

    def forward(self, x):

        if self.convx is None:
            conv_out = F.leaky_relu(self.encode(x) + x)
        else:
            conv_out = F.leaky_relu(self.encode(x) + self.convx(x))

        conv_out = self.csa(conv_out)

        down_out = F.max_pool2d(conv_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return conv_out, down_out


class StackResCenterCSA2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackResCenterCSA2d, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False),
        )

        self.csa = CBAM(out_chl)

        self.convx = None

        if in_chl != out_chl:
            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                      is_relu=False)

    def forward(self, x):

        if self.convx is None:
            conv_out = F.leaky_relu(self.encode(x) + x)
        else:
            conv_out = F.leaky_relu(self.encode(x) + self.convx(x))

        conv_out = self.csa(conv_out)

        return conv_out


class ResUNet2dSCA(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super(ResUNet2dSCA, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackResEncoderSCA2d(self.model_chl, self.model_chl, kernel_size=3)  # 256
        self.down2 = StackResEncoderSCA2d(self.model_chl * 1, self.model_chl * 2, kernel_size=3)  # 128
        self.down3 = StackResEncoderSCA2d(self.model_chl * 2, self.model_chl * 4, kernel_size=3)  # 64
        self.down4 = StackResEncoderSCA2d(self.model_chl * 4, self.model_chl * 8, kernel_size=3)  # 32

        self.center = StackResCenterCSA2d(self.model_chl * 8, self.model_chl * 16, kernel_size=3)  # 32

        self.up4 = StackResDecoder2d(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackResDecoder2d(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackResDecoder2d(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackResDecoder2d(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(
            ConvBnRelu2d(self.model_chl, self.out_chl, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)
        conv2, d2 = self.down2(d1)
        conv3, d3 = self.down3(d2)
        conv4, d4 = self.down4(d3)
        conv5 = self.center(d4)
        up4 = self.up4(conv5, conv4)
        up3 = self.up3(up4, conv3)
        up2 = self.up2(up3, conv2)
        up1 = self.up1(up2, conv1)
        conv6 = self.end(up1)
        res_out = F.leaky_relu(x + conv6)
        return res_out