from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_mode, stride = 1, padding = 0, bias = False):
        super(RotationConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_mode = kernel_mode
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.kernel_mode == '3D':
            self.kernel = nn.Parameter(torch.Tensor(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size
            ))
            self.kernel.data.normal_(0, 0.02)
        elif self.kernel_mode == '2D+1D':
            self.kernel_2d = nn.Parameter(torch.Tensor(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
            ))
            self.kernel_2d.data.normal_(0, 0.02)

            self.kernel_1d = nn.Parameter(torch.Tensor(
                self.out_channels, self.in_channels, self.kernel_size
            ))
            self.kernel_1d.data.fill_(1)
        else:
            raise AssertionError('unsupported kernel mode "{0}"'.format(self.kernel_mode))

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.bias = None

    def forward(self, inputs):
        if self.kernel_mode == '3D':
            kernel = self.kernel

        if self.kernel_mode == '2D+1D':
            i, o, k = self.in_channels, self.out_channels, self.kernel_size

            s = int(o / 3)
            assert s > 0

            kernel_x = self.kernel_2d[:s].view(s, i, 1, k, k) * \
                self.kernel_1d[:s].view(s, i, k, 1, 1)
            kernel_y = self.kernel_2d[s:s * 2].view(s, i, k, 1, k) * \
                self.kernel_1d[s:s * 2].view(s, i, 1, k, 1)
            kernel_z = self.kernel_2d[s * 2:].view(-1, i, k, k, 1) * \
                self.kernel_1d[s * 2:].view(-1, i, 1, 1, k)
            kernel = torch.cat([kernel_x, kernel_y, kernel_z], 0)

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
