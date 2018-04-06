import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch import to_var, transform_grid


def transform3d(theta):
    theta_n = theta[:, :-1]
    theta_r = theta[:, -1]

    normal = F.normalize(theta_n, p = 2)

    transform = to_var(torch.zeros(theta.size(0), 3, 3))
    transform[:, 2, 1], transform[:, 1, 2] = normal[:, 0], -normal[:, 0]
    transform[:, 0, 2], transform[:, 2, 0] = normal[:, 1], -normal[:, 1]
    transform[:, 1, 0], transform[:, 0, 1] = normal[:, 2], -normal[:, 2]

    theta = to_var(torch.eye(3)).view(1, 3, 3) + \
            torch.sin(theta_r).view(-1, 1, 1) * transform + \
            (1 - torch.cos(theta_r)).view(-1, 1, 1) * torch.bmm(transform, transform)

    return theta





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
            self.theta_n = nn.Parameter(torch.zeros(self.out_channels * self.in_channels, 3))
            nn.init.uniform(self.theta_n, a = 0, b = 1)

            self.theta_r = nn.Parameter(torch.zeros(self.out_channels * self.in_channels))
            nn.init.uniform(self.theta_r, a = 0, b = np.pi)

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
            theta = torch.cat([self.theta_n, self.theta_r.unsqueeze(-1)], -1)

            transform = transform3d(theta)
            grids = transform_grid(transform, (i * o, 1, k, k, k))

            kernels = kernels.view(-1, 1, k, k, k)
            kernels = F.grid_sample(kernels, grids)
            kernels = kernels.view(o, i, k, k, k)

        outputs = F.conv3d(inputs, kernels, bias = self.bias, stride = self.stride, padding = self.padding)
        return outputs

