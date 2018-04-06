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
            self.kernels = [
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size
                ))
            ]

        if '2d+1d' in self.kernel_mode:
            self.kernels = [
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
                )),
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size
                ))
            ]

        if '1d+1d+1d' in self.kernel_mode:
            self.kernels = [
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size
                )),
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size
                )),
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size
                ))
            ]

        for k, kernel in enumerate(self.kernels):
            nn.init.normal(kernel, std = 0.02)
            self.register_parameter('kernel-{0}'.format(k + 1), kernel)

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
        i, o, k = self.in_channels, self.out_channels, self.kernel_size

        kernels = to_var(torch.ones(o * i))
        for kernel in self.kernels:
            kernels = torch.bmm(
                kernels.view(o * i, -1, 1),
                kernel.view(o * i, 1, -1)
            )
        kernels = kernels.view(o, i, k, k, k)

        if 'rot' in self.kernel_mode:
            kernels = self.rotate3d.forward(kernels)

        outputs = F.conv3d(inputs, kernels, bias = self.bias, stride = self.stride, padding = self.padding)
        return outputs


class Rotate3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Rotate3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.base_grids = to_var(torch.zeros(self.kernel_size, self.kernel_size, self.kernel_size, 3))
        for k in range(self.kernel_size):
            self.base_grids[:, :, k, 0] = k * 2. / (self.kernel_size - 1) - 1
            self.base_grids[:, k, :, 1] = k * 2. / (self.kernel_size - 1) - 1
            self.base_grids[k, :, :, 2] = k * 2. / (self.kernel_size - 1) - 1

        self.theta_n = nn.Parameter(torch.zeros(self.out_channels * self.in_channels, 3))
        nn.init.uniform(self.theta_n, a = 0, b = 1)

        self.theta_r = nn.Parameter(torch.zeros(self.out_channels * self.in_channels))
        nn.init.uniform(self.theta_r, a = 0, b = np.pi)

    def forward(self, inputs):
        o, i, k = inputs.size(0), inputs.size(1), inputs.size(2)

        normal = F.normalize(self.theta_n, p = 2)

        transform = to_var(torch.zeros(o * i, 3, 3))
        transform[:, 2, 1], transform[:, 1, 2] = normal[:, 0], -normal[:, 0]
        transform[:, 0, 2], transform[:, 2, 0] = normal[:, 1], -normal[:, 1]
        transform[:, 1, 0], transform[:, 0, 1] = normal[:, 2], -normal[:, 2]

        theta = to_var(torch.eye(3)).view(1, 3, 3) + \
                torch.sin(self.theta_r).view(-1, 1, 1) * transform + \
                (1 - torch.cos(self.theta_r)).view(-1, 1, 1) * torch.bmm(transform, transform)

        inputs = inputs.view(-1, 1, k, k, k)

        grids = torch.stack([self.base_grids] * (self.out_channels * self.in_channels))
        grids = grids.view(-1, k * k * k, 3)
        grids = torch.bmm(grids, theta)
        grids = grids.view(-1, k, k, k, 3)

        outputs = F.grid_sample(inputs, grids)
        outputs = outputs.view(o, i, k, k, k)
        return outputs
