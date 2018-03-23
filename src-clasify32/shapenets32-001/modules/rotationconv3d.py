from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class RotationConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(RotationConv3d, self).__init__()
        if not isinstance(kernel_size, int):
            print("RotationConv3d only support k*k*k size for now")
            exit(1)

        self.o_c = out_channels
        self.i_c = in_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        ### hidden states
        self.weight = nn.Parameter(torch.Tensor(self.o_c,self.i_c,self.k,self.k))
        
        self.weight_1d = nn.Parameter(torch.Tensor(self.o_c,self.i_c,self.k))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.o_c))
        else:
            self.bias = None

    def forward(self, input):
        filter = self._get_filter(self.weight, self.weight_1d)
        return nn.functional.conv3d(input, filter, bias=self.bias, stride=self.stride, padding=self.padding)

    def _get_filter(self, weight, weight_1d):
        """
        weight: (out,in,k,k)
        weight_1d: (out,in,k)
        filter: (out,in,k,k,k)
        """
        o_c, i_c, k, _ = weight.size()
        sub_out = int(o_c / 3)
        assert(sub_out != 0)

        filter_x = weight[:sub_out].view(sub_out,i_c,1,k,k) * weight_1d[:sub_out].view(sub_out,i_c,k,1,1)
        filter_y = weight[sub_out:sub_out*2].view(sub_out,i_c,k,1,k) * weight_1d[sub_out:sub_out*2].view(sub_out,i_c,1,k,1)
        filter_z = weight[sub_out*2:].view(-1,i_c,k,k,1) * weight_1d[sub_out*2:].view(-1,i_c,1,1,k)

        filter = torch.cat((filter_x, filter_y, filter_z), dim=0)
        return filter

    def __repr__(self):
        s = self.__class__.__name__
        s += '('
        s += 'in=%d, '%(self.i_c)
        s += 'out=%d, '%(self.o_c)
        s += 'kernel_size=%s, '%(self.k)
        s += 'stride=%d, '%(self.stride)
        s += 'padding=%d, '%(self.padding)
        s += 'bias=%d, '%(self.bias is None)
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)



