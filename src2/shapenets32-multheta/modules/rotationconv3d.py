from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import numpy as np

from utils.modules import BasicRotation, FastRotation
from utils.torchhelper import to_np, to_var, torchmax


class RotationConv3d(nn.Module):
    def __init__(self, kernel_mode, num_theta, in_channels, out_channels, 
            kernel_size, stride=1, padding=0, bias=False):
        super(RotationConv3d, self).__init__()
        if isinstance(kernel_size, int):
            kT, kH, kW = kernel_size, kernel_size, kernel_size
        else:
            kT, kH, kW = kernel_size
        self.kernel_mode = kernel_mode
        self.num_theta = num_theta
        self.i_c = in_channels
        self.o_c = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding  

        self.weight = nn.Parameter(torch.Tensor(self.o_c,self.i_c,self.k,self.k,self.k))
        self.weight.data.normal_(0, 0.02)     

        if 'rot' in  self.kernel_mode:
            n = self.i_c * self.o_c
            self.theta_v = nn.Parameter(torch.Tensor(n, 3))    # (n, 3)
            self.theta = nn.Parameter(torch.Tensor(n))         # (n)
            self.theta_v.data.uniform_(0,1)
            self.theta.data.uniform_(0,np.pi)

            self.rotation = FastRotation(n, self.k, padding_mode='zeros')


        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None


    def forward(self, input):
        filter = self.weight
        if 'rot' in self.kernel_mode:
            output_list = []
            for i in range(self.num_theta):
                extra_alpha = i * 2*np.pi / self.num_theta
                c_filter = self.rotation(filter, self.theta_v, self.theta, extra_alpha=extra_alpha)

                c_output = nn.functional.conv3d(
                    input, c_filter, bias=self.bias, stride=self.stride, padding=self.padding
                )
                output_list.append(c_output)
            output = torchmax(output_list)
        else:
            output = nn.functional.conv3d(
                input, filter, bias=self.bias, stride=self.stride, padding=self.padding
            )
        return output 


    def __repr__(self):
        s = '{name}('
        s += ', kernel_mode={kernel_mode}'
        s += ', in={i_c}'
        s += ', out={o_c}'
        s += ', kernel_size={k}'
        s += ', stride={stride}'
        s += ', padding={padding}'
        s += ', bias={bias}'
        s += ')'
        return (s.format(name=self.__class__.__name__, **self.__dict__))





