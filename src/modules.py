from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

def registered_kernels():
    kernel_dict = {}
    current_module = sys.modules[__name__]
    for key in dir(current_module):
        if isinstance(getattr(current_module, key), type):
            kernel = getattr(current_module, key)
            if callable(getattr(kernel, 'registered_name', None)):
                name = kernel.registered_name()
                if name in kernel_dict:
                    raise AssertionError('Name %s already exists'%(name))
                kernel_dict[name] = kernel
    return kernel_dict



class PlainConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, bias = False):
        super(PlainConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.kernel = nn.Parameter(torch.Tensor(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size
        ))
        self.kernel.data.normal_(0, 0.02)

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.bias = None

    def forward(self, inputs):
        kernel = self.kernel
        outputs = F.conv3d(inputs, kernel, bias = self.bias, stride = self.stride, padding = self.padding)
        return outputs

    @staticmethod
    def registered_name():
        return "3D"


class Conv2d_1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, bias = False):
        super(Conv2d_1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.kernel_2d = nn.Parameter(torch.Tensor(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        ))
        self.kernel_2d.data.normal_(0, 0.02)

        self.kernel_1d = nn.Parameter(torch.Tensor(
            self.out_channels, self.in_channels, self.kernel_size
        ))
        self.kernel_1d.data.fill_(1)

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.bias = None

    def forward(self, inputs):
        i, o, k = self.in_channels, self.out_channels, self.kernel_size
        kernel_2d = self.kernel_2d.view(o * i, k * k, 1)
        kernel_1d = self.kernel_1d.view(o * i, 1, k)
        kernel = torch.bmm(kernel_2d, kernel_1d).view(o, i, k, k, k)

        outputs = F.conv3d(inputs, kernel, bias = self.bias, stride = self.stride, padding = self.padding)
        return outputs

    @staticmethod
    def registered_name():
        return "2D+1D"









