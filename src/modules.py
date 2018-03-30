from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch import to_var


class ConvRotate3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_mode, stride = 1, padding = 0, bias = True):
        super(ConvRotate3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_mode = kernel_mode
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if '3d' in self.kernel_mode:
            self.weights = nn.Parameter(torch.zeros(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size
            ))
            nn.init.normal(self.weights, std = 0.02)

        if '2d+1d' in self.kernel_mode:
            self.weights_2d = nn.Parameter(torch.zeros(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
            ))
            nn.init.normal(self.weights_2d, std = 0.02)

            self.weights_1d = nn.Parameter(torch.zeros(
                self.out_channels, self.in_channels, self.kernel_size
            ))
            nn.init.normal(self.weights_1d, std = 0.02)

        if 'rot' in self.kernel_mode:
            self.rotate3d = Rotate3d(
                in_channels = self.in_channels,
                out_channels = self.out_channels,
                kernel_size = self.kernel_size
            )

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.bias = None

    def forward(self, inputs):
        if '3d' in self.kernel_mode:
            weights = self.weights

        if '2d+1d' in self.kernel_mode:
            i, o, k = self.in_channels, self.out_channels, self.kernel_size
            weights_2d = self.weights_2d.view(o * i, k * k, 1)
            weights_1d = self.weights_1d.view(o * i, 1, k)
            weights = torch.bmm(weights_2d, weights_1d).view(o, i, k, k, k)

        if 'rot' in self.kernel_mode:
            weights = self.rotate3d.forward(weights)

        outputs = F.conv3d(inputs, weights, bias = self.bias, stride = self.stride, padding = self.padding)
        return outputs


class Rotate3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Rotate3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.base_grids = to_var(torch.zeros(
            self.out_channels * self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size, 3
        ))
        for k in range(self.kernel_size):
            self.base_grids[:, :, :, k, 0] = k * 2. / (self.kernel_size - 1) - 1
            self.base_grids[:, :, k, :, 1] = k * 2. / (self.kernel_size - 1) - 1
            self.base_grids[:, k, :, :, 2] = k * 2. / (self.kernel_size - 1) - 1

        self.theta_v = nn.Parameter(torch.zeros(self.out_channels * self.in_channels, 3))
        nn.init.uniform(self.theta_v, a = 0, b = 1)

        self.theta = nn.Parameter(torch.zeros(self.out_channels * self.in_channels))
        nn.init.uniform(self.theta, a = 0, b = np.pi)

    def forward(self, inputs):
        o, i, k = inputs.size(0), inputs.size(1), inputs.size(2)

        inputs = inputs.view(-1, 1, k, k, k)

        theta_v = F.normalize(self.theta_v, p = 2)
        vx = theta_v[:, 0]
        vy = theta_v[:, 1]
        vz = theta_v[:, 2]

        m = to_var(torch.zeros(o * i, 3, 3))
        m[:, 0, 1] = -vz
        m[:, 0, 2] = vy
        m[:, 1, 0] = vz
        m[:, 1, 2] = -vx
        m[:, 2, 0] = -vy
        m[:, 2, 1] = vx

        I3 = to_var(torch.eye(3)).view(1, 3, 3)
        R = I3 + torch.sin(self.theta).view(-1, 1, 1) * m + (1 - torch.cos(self.theta)).view(-1, 1, 1) * torch.bmm(m, m)

        grids = torch.bmm(self.base_grids.view(-1, k * k * k, 3), R)
        grids = grids.view(-1, k, k, k, 3)

        outputs = F.grid_sample(inputs, grids)
        outputs = outputs.view(o, i, k, k, k)
        return outputs
