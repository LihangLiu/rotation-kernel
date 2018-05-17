import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils.modules import BasicRotation, FastRotation
from utils.torchhelper import to_np, to_tensor, to_var, torchmax

class RotationConv3d(nn.Module):
    def __init__(self, kernel_mode, in_channels, out_channels, 
            kernel_size, stride=1, padding=0, bias=False, num_theta=2):
        super(RotationConv3d, self).__init__()
        if isinstance(kernel_size, int):
            kT, kH, kW = kernel_size, kernel_size, kernel_size
        else:
            kT, kH, kW = kernel_size
        self.kernel_mode = kernel_mode
        self.i_c = in_channels
        self.o_c = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding  
        self.num_theta = num_theta

        if '3d' in self.kernel_mode:
            self.weight = nn.Parameter(torch.Tensor(self.o_c,self.i_c,self.k,self.k,self.k))
            self.weight.data.normal_(0, 0.02)

        elif '2d_1d' in self.kernel_mode:
            self.weight_2d = nn.Parameter(torch.Tensor(self.o_c,self.i_c,self.k,self.k))
            self.weight_1d = nn.Parameter(torch.Tensor(self.o_c,self.i_c,self.k))
            self.weight_2d.data.normal_(0, 0.02)
            self.weight_1d.data.normal_(0, 0.02)

        else:
            raise NotImplementedError(kernel_mode)
            
        if 'rot' in self.kernel_mode:
            m = self.i_c * self.o_c
            self.local_theta_v = nn.Parameter(torch.Tensor(m, 3))    # (n, 3)
            self.local_theta = nn.Parameter(torch.Tensor(m))         # (n)
            self.local_theta_v.data.uniform_(-1,1)
            self.local_theta.data.uniform_(0,np.pi)

            self.local_rotation = FastRotation(m, self.k, padding_mode='zeros')

        if 'mn' in self.kernel_mode:
            m = self.i_c * self.o_c
            n = self.num_theta
            self.theta_v = nn.Parameter(torch.Tensor(num_theta, 3))    # (n, 3)
            self.theta = nn.Parameter(torch.Tensor(num_theta))         # (n)
            if 'init1' in self.kernel_mode:
                self.theta_init_1(self.theta_v, self.theta)
            elif 'init2' in self.kernel_mode:
                self.theta_init_2(self.theta_v, self.theta)
            elif 'init3' in self.kernel_mode:
                self.theta_init_3(self.theta_v, self.theta)
            elif 'init4' in self.kernel_mode:
                del self.theta_v, self.theta
                self.theta_v = to_var(torch.zeros(4, 3))
                self.theta = to_var(torch.arange(4)*np.pi/2)
                self.theta_v.data[:, 2] = 1.0

            # self.rotation = FastRotation(m, self.k, padding_mode='zeros')

            # try fast
            self.rotation = FastRotation(n*m, self.k, padding_mode='zeros')


        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

    def theta_init_1(self, theta_v, theta):
        """
        random init
        """
        theta_v.data.uniform_(-1,1)
        theta.data.uniform_(-np.pi,np.pi)

    def theta_init_2(self, theta_v, theta):
        """
        along z
        """
        n = theta_v.size(0)
        theta_v.data.fill_(0.0)
        theta_v.data[:, 2] = 1.0
        theta.data.uniform_(-np.pi,np.pi)

    def theta_init_3(self, theta_v, theta):
        """
        along y
        """
        n = theta_v.size(0)
        theta_v.data.fill_(0.0)
        theta_v.data[:, 1] = 1.0
        theta.data.uniform_(-np.pi,np.pi)

    def forward(self, input):
        filter = self._get_filter()
        
        if 'rot' in self.kernel_mode:
            filter = self.local_rotation(filter, self.local_theta_v, self.local_theta)
        
        if 'mn' in self.kernel_mode:
            # m = self.i_c * self.o_c
            # n = self.num_theta
            # out_list = []
            # for i in range(n):
            #     c_theta_v, c_theta = self.replicate_theta(self.theta_v[i:i+1], self.theta[i:i+1], m)
            #     c_filter = self.rotation(filter, c_theta_v, c_theta)
            #     c_output = nn.functional.conv3d(input, c_filter, bias=self.bias, stride=self.stride, padding=self.padding)
            #     out_list.append(c_output)
            # output = torch.cat(out_list, dim=1)
            # return output

            # try fast
            m = self.i_c * self.o_c
            n = self.num_theta
            filter, theta_v, theta = self.replicate_filter_and_theta(filter, self.theta_v, self.theta)
            # print(filter.size(), theta_v.size(), theta.size())
            rot_filter = self.rotation(filter, theta_v, theta)
            output = nn.functional.conv3d(input, rot_filter, bias=self.bias, stride=self.stride, padding=self.padding)
            return output

        else:
            return nn.functional.conv3d(input, filter, bias=self.bias, stride=self.stride, padding=self.padding)

    def _get_filter(self):
        if '3d' in self.kernel_mode:
            filter = self.weight

        elif '2d_1d' in self.kernel_mode:
            o, i, k, _ = self.weight_2d.size()
            weight_2d = self.weight_2d.view(o*i, k*k, 1)
            weight_1d = self.weight_1d.view(o*i, 1, k)
            filter = torch.bmm(weight_2d, weight_1d).view(o,i,k,k,k)

        return filter

    def replicate_theta(self, theta_v, theta, m):
        """
        theta_v: (n, 3)
        theta: (n)
        return:
            theta_v: (n*m, 3)
            theta: (n*m)
        """
        # don't use repeat() as it's super slow

        n = theta_v.size(0)
        new_theta_v = theta_v.view(n, 1, 3) + to_var(torch.zeros(1,m,1))
        new_theta = theta.view(n, 1) + to_var(torch.zeros(1, m))
        return new_theta_v.view(n*m, 3), new_theta.view(n*m)

    def replicate_filter_and_theta(self, filter, theta_v, theta):
        """
        filter: (o, i, k, k, k)
        theta_v: (n, 3)
        theta: (n)
        """
        o, i, k, _, _ = filter.size()
        m = o*i
        n = theta_v.size(0)

        # (o, i) -> (n*o, i)
        filter = filter.view(1,o,i,k,k,k) + to_var(torch.zeros(n,1,1,1,1,1))
        filter = filter.view(n*o,i,k,k,k)

        # (n, 3) -> (n*m, 3)
        theta_v = theta_v.view(n, 1, 3) + to_var(torch.zeros(1,m,1))
        theta_v = theta_v.view(n*m, 3)
        theta = theta.view(n, 1) + to_var(torch.zeros(1, m))
        theta = theta.view(n*m)
        return filter, theta_v, theta

    def __repr__(self):
        s = '{name}('
        s += ', kernel_mode={kernel_mode}'
        s += ', in={i_c}'
        s += ', out={o_c}'
        s += ', kernel_size={k}'
        s += ', stride={stride}'
        s += ', padding={padding}'
        s += ', bias={bias}'
        s += ', num_theta={num_theta}'
        s += ')'
        return (s.format(name=self.__class__.__name__, **self.__dict__))







