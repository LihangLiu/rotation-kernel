from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class RotationConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(RotationConv3d, self).__init__()
        if isinstance(kernel_size, int):
            kT, kH, kW = kernel_size, kernel_size, kernel_size
        else:
            kT, kH, kW = kernel_size
        ### full 3d
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,kT,kH,kW))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        filter = self._get_filter(self.weight)
        return nn.functional.conv3d(input, filter, bias=self.bias, stride=self.stride, padding=self.padding)

    def _get_filter(self, weight):
        return weight

    def __repr__(self):
        s = self.__class__.__name__
        s += '('
        s += 'in=%d, '%(self.in_channels)
        s += 'out=%d, '%(self.out_channels)
        s += 'kernel_size=%d, '%(self.kernel_size)
        s += 'stride=%d, '%(self.stride)
        s += 'padding=%d, '%(self.padding)
        s += 'bias=%d, '%(self.bias is None)
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


        
