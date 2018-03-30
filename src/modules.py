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
            self.theta_v = nn.Parameter(torch.zeros(self.out_channels * self.in_channels, 3))  # (o*i, 3)
            nn.init.uniform(self.theta_v, a = 0, b = 1)

            self.theta = nn.Parameter(torch.zeros(self.out_channels * self.in_channels))  # (o*i)
            nn.init.uniform(self.theta, a = 0, b = np.pi)

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
            weights = rotate(weights, self.theta_v, self.theta)

        outputs = F.conv3d(inputs, weights, bias = self.bias, stride = self.stride, padding = self.padding)
        return outputs


def rotate(inputs, theta_v, theta):
    i, o, k = inputs.size(0), inputs.size(1), inputs.size(2)

    indices = np.zeros((k, k, k, 3))
    for x in range(k):
        for y in range(k):
            for z in range(k):
                indices[x, y, z, 0] = z * 2. / (k - 1) - 1
                indices[x, y, z, 1] = y * 2. / (k - 1) - 1
                indices[x, y, z, 2] = x * 2. / (k - 1) - 1
    indices = np.stack([indices] * (o * i))

    # base_grid = to_var(np.zeros((o * i, k, k, k, 3)))
    #
    # linear_points = torch.linspace(-1, 1, k)
    # base_grid[:, :, :, 0] = torch.ger(torch.ones(k), linear_points).expand_as(base_grid[:, :, :, 0])
    #
    # linear_points = torch.linspace(-1, 1, k)
    # base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(k)).expand_as(base_grid[:, :, :, 1])
    #
    # base_grid[:, :, :, 2] = 1

    inputs = inputs.view(-1, 1, k, k, k)

    base_grid = to_var(indices)
    n = theta_v.size(0)

    theta_v = F.normalize(theta_v, p = 2)
    vx = theta_v[:, 0]
    vy = theta_v[:, 1]
    vz = theta_v[:, 2]

    m = to_var(torch.zeros(n, 3, 3))
    m[:, 0, 1] = -vz
    m[:, 0, 2] = vy
    m[:, 1, 0] = vz
    m[:, 1, 2] = -vx
    m[:, 2, 0] = -vy
    m[:, 2, 1] = vx

    I3 = to_var(torch.eye(3)).view(1, 3, 3)
    R = I3 + torch.sin(theta).view(n, 1, 1) * m + (1 - torch.cos(theta)).view(n, 1, 1) * torch.bmm(m, m)

    grids = torch.bmm(base_grid.view(-1, k * k * k, 3), R)
    grids = grids.view(-1, k, k, k, 3)

    outputs = F.grid_sample(inputs, grids)
    outputs = outputs.view(o, i, k, k, k)
    return outputs
