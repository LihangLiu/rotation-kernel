from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import numpy as np

class RotationConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(RotationConv3d, self).__init__()
        if not isinstance(kernel_size, int):
            print("RotationConv3d only support k*k*k size for now")
            exit(1)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

        ### full 3d
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,self.k,self.k,self.k))

        self.weight_2d = nn.Parameter(torch.Tensor(out_channels,in_channels,self.k,self.k))
        self.weight_1d = nn.Parameter(torch.Tensor(out_channels,in_channels,self.k))

        self.use_decompose = nn.Parameter(torch.zeros(1), requires_grad=False)  # init as False

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

    def forward(self, input):
        filter = self._get_filter()
        return nn.functional.conv3d(input, filter, bias=self.bias, stride=self.stride, padding=self.padding)

    def enable_3d_decompose(self):
        o, i, k, _, _ = self.weight.size()
        print('Decompose (', o, i, k, k, k, ')...')
        self.use_decompose.data[0] = 1

        weight = self.weight.data.view(o*i, k*k, k)
        weight_2d = self.weight_2d.data.view(o*i, k*k, 1)
        weight_1d = self.weight_1d.data.view(o*i, 1, k)
        for n in range(o*i):
            u, s, v = torch.svd(weight[n])
            weight_2d[n] = u[:,0].contiguous()*np.sqrt(s[0])
            weight_1d[n] = v[:,0].contiguous()*np.sqrt(s[0])

        self.weight_2d.data[:] = weight_2d.view(o,i,k,k)
        self.weight_1d.data[:] = weight_1d.view(o,i,k)

    def stat_svd_s(self):
        o, i, k, _, _ = self.weight.size()
        res = []
        weight = self.weight.data.view(o*i, k*k, k)
        for n in range(o*i):
            u, s, v = torch.svd(weight[n])
            res.append(s.cpu().numpy())
        return np.array(res)

    def _get_filter(self):
        o, i, k, _, _ = self.weight.size()
        if self.use_decompose.data[0] == 1:
            weight_2d = self.weight_2d.view(o*i, k*k, 1)
            weight_1d = self.weight_1d.view(o*i, 1, k)
            filter = torch.bmm(weight_2d, weight_1d).view(o,i,k,k,k)
        else:
            filter = self.weight
        return filter

    def __repr__(self):
        s = self.__class__.__name__
        s += '('
        s += 'in=%d, '%(self.in_channels)
        s += 'out=%d, '%(self.out_channels)
        s += 'kernel_size=%d, '%(self.k)
        s += 'stride=%d, '%(self.stride)
        s += 'padding=%d, '%(self.padding)
        s += 'bias=%d, '%(self.bias is None)
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


        
