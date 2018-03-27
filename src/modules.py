from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRotate3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_mode, stride = 1, padding = 0, bias = False):
        super(ConvRotate3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_mode = kernel_mode
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.kernel_mode == '3d':
            self.kernel = nn.Parameter(torch.Tensor(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size
            ))
            nn.init.normal(self.kernel, std = 0.02)

        if self.kernel_mode == '2d+1d':
            self.kernel_2d = nn.Parameter(torch.Tensor(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
            ))
            nn.init.normal(self.kernel_2d, std = 0.02)

            self.kernel_1d = nn.Parameter(torch.Tensor(
                self.out_channels, self.in_channels, self.kernel_size
            ))
            nn.init.normal(self.kernel_1d, std = 0.02)

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.bias = None

    def forward(self, inputs):
        if self.kernel_mode == '3d':
            kernel = self.kernel

        if self.kernel_mode == '2d+1d':
            i, o, k = self.in_channels, self.out_channels, self.kernel_size
            kernel_2d = self.kernel_2d.view(o * i, k * k, 1)
            kernel_1d = self.kernel_1d.view(o * i, 1, k)
            kernel = torch.bmm(kernel_2d, kernel_1d).view(o, i, k, k, k)

        outputs = F.conv3d(inputs, kernel, bias = self.bias, stride = self.stride, padding = self.padding)
        return outputs

    # def __repr__(self):
    #     s = self.__class__.__name__
    #     s += '('
    #     s += 'in=%d, ' % (self.in_channels)
    #     s += 'out=%d, ' % (self.out_channels)
    #     s += 'kernel_size=%d, ' % (self.kernel_size)
    #     s += 'stride=%d, ' % (self.stride)
    #     s += 'padding=%d, ' % (self.padding)
    #     s += 'bias=%d, ' % (self.bias is None)
    #     s += ')'
    #     return s.format(name = self.__class__.__name__, **self.__dict__)
