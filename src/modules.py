from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
            self.rotation = Rotation(
                self.in_channels, self.out_channels, self.kernel_size
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
            weights = self.rotation.forward(weights)

        outputs = F.conv3d(inputs, weights, bias = self.bias, stride = self.stride, padding = self.padding)
        return outputs


class Rotation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Rotation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.theta_v = nn.Parameter(torch.zeros(self.out_channels * self.in_channels, 3))  # (o*i, 3)
        self.theta = nn.Parameter(torch.zeros(self.out_channels * self.in_channels))  # (o*i)

        ### some constants used to calc C
        self.const_3d_indice_xyz = self._get_3d_indice(self.kernel_size).cuda().unsqueeze(0).repeat(
            self.out_channels * self.in_channels, 1, 1)  # (o*i, k^3, 3)

        self.theta_v.data.uniform_(0, 1)
        self.theta.data.uniform_(0, np.pi)

    def forward(self, input_filter):
        assert input_filter.size(0) == self.out_channels \
               and input_filter.size(1) == self.in_channels \
               and input_filter.size(2) == self.kernel_size

        input_filter = input_filter.view(-1, self.kernel_size, self.kernel_size, self.kernel_size)
        filter = self._linear_interpolation(self.theta_v, self.theta, input_filter)
        filter = filter.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size)
        return filter

    def _linear_interpolation(self, theta_v, theta, weight_3d):
        """
        theta_v: (n, 3)
        theta: (n,)
        weight_3d: (n, k, k, k)
        return:
            filter: (n, k, k, k)
        """
        n = theta_v.size(0)
        k = self.kernel_size

        ### o_c*i_c kernels needed
        ###########################
        ### get Rotation matrix
        ###########################
        R = self._get_R_by_expm(theta_v, theta)

        ###########################
        ### rotate the 3d kernel
        ###########################
        indice = Variable(self.const_3d_indice_xyz)
        # translate
        indice = indice - (k - 1) / 2.0
        new_indice = torch.bmm(indice, R)  # (n, k^3, 3)
        # translate back
        new_indice = new_indice + (k - 1) / 2.0

        ######################################
        # interpolate from the weight to the filter
        ######################################
        delta = torch.Tensor([[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0],
                              [0, 1, 1],
                              [1, 0, 0],
                              [1, 0, 1],
                              [1, 1, 0],
                              [1, 1, 1]]
                             ).cuda()
        batch_indice = torch.arange(0, n).cuda().long().view(n, 1).repeat(1, k ** 3).view(n * k ** 3)  # (n*k^3,)
        xyz = new_indice.view(n * k ** 3, 3)  # (n*k^3, 3)
        filter = 0
        for i in range(8):
            indice_xyz = Variable(xyz.data.floor() + delta[i:i + 1])  # (n*k^3, 3)
            valid_xyz = (indice_xyz >= 0) * (indice_xyz <= k - 1)
            indice_x = indice_xyz[:, 0]
            indice_y = indice_xyz[:, 1]
            indice_z = indice_xyz[:, 2]
            c_filter = weight_3d[batch_indice, torch.clamp(indice_x, 0, k - 1).long(), torch.clamp(indice_y, 0,
                                                                                                   k - 1).long(), torch.clamp(
                indice_z, 0, k - 1).long()]  # (n*k^3,)
            c_filter *= (1 - torch.abs(xyz[:, 0] - indice_x)) * (1 - torch.abs(xyz[:, 1] - indice_y)) * (
                    1 - torch.abs(xyz[:, 2] - indice_z))
            c_filter *= valid_xyz[:, 0].float() * valid_xyz[:, 1].float() * valid_xyz[:, 2].float()

            filter += c_filter

        filter = filter.view(n, k, k, k)

        return filter

    def _get_R_by_expm(self, v, theta):
        """
        v: (o*i, 3)
        theta: (o*i,)
        return:
            R: (o*i, 3, 3)    
        """
        n = v.size(0)

        # normalize v
        epsilon = 0.000000001  # used to handle 0/0
        v_length = torch.sqrt(torch.sum(v * v, dim = 1))
        vx = (v[:, 0] + epsilon) / (v_length + epsilon)
        vy = (v[:, 1] + epsilon) / (v_length + epsilon)
        vz = (v[:, 2] + epsilon) / (v_length + epsilon)

        m = Variable(torch.zeros(n, 3, 3)).cuda()
        m[:, 0, 1] = -vz
        m[:, 0, 2] = vy
        m[:, 1, 0] = vz
        m[:, 1, 2] = -vx
        m[:, 2, 0] = -vy
        m[:, 2, 1] = vx

        I3 = Variable(torch.eye(3).view(1, 3, 3).cuda())
        R = I3 + torch.sin(theta).view(n, 1, 1) * m + (1 - torch.cos(theta)).view(n, 1, 1) * torch.bmm(m, m)
        return R

    def _get_3d_indice(self, k):
        """
        indice: (k^3, 3)
        """
        a = torch.arange(0, k ** 3).long()
        indice = torch.stack([a / (k * k), a % (k * k) / k, a % (k)], dim = 1).float()  # (k^3, 3)
        return indice
