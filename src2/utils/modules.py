import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .torchhelper import to_var

class ForceRegularization(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        k = input.size(2)
        loss = torch.zeros([]).cuda()
        for kk in range(k):
            loss += torch.mean(torch.pow((input[:,:,kk:kk+1] - input), 2))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]

        n, _, k = input.size()
        if ctx.needs_input_grad[0]:
            grad_input = []
            for i in range(k):
                gradi = 0
                wi = input[:,:,i:i+1].contiguous()   # (n, k^2, 1)
                wiTwi = torch.bmm(wi, wi.view(n, 1, k*k))
                for j in range(k):
                    fji = input[:,:,j:j+1] - wi
                    gradi += (fji - torch.bmm(wiTwi,fji))   # (n, k^2, 1)
                grad_input.append(gradi)
            grad_input = torch.cat(grad_input, dim=2)       # (n, k^2, k)

        return - grad_output * grad_input

class SparsityRegularization(nn.Module):
    def __init__(self):
        super(SparsityRegularization, self).__init__()

    def forward(ctx, input):
        return torch.mean(torch.abs(input))



class BasicRotation(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super(BasicRotation, self).__init__()
        self.n = num_channels
        self.k = kernel_size
        
        ### some constants used to calc C
        self.const_3d_indice_xyz = self._get_3d_indice(self.k).cuda().unsqueeze(0).repeat(self.n,1,1)  # (o*i, k^3, 3)

    def forward(self, input_filter, theta_v, theta):
        filter = input_filter.view(-1,self.k,self.k,self.k)
        assert filter.size(0) == self.n

        filter = self._linear_interpolation(theta_v, theta, filter)
        filter = filter.view(input_filter.size())
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
        k = self.k

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
        indice = indice - (k-1)/2.0
        new_indice = torch.bmm(indice, R)                   # (n, k^3, 3)
        # translate back
        new_indice = new_indice + (k-1)/2.0

        ######################################
        # interpolate from the weight to the filter
        ######################################
        delta = torch.Tensor([[0,0,0],
                                  [0,0,1],
                                  [0,1,0],
                                  [0,1,1],
                                  [1,0,0],
                                  [1,0,1],
                                  [1,1,0],
                                  [1,1,1]]
                                ).cuda()
        batch_indice = torch.arange(0,n).cuda().long().view(n,1).repeat(1,k**3).view(n*k**3)  # (n*k^3,)
        xyz = new_indice.view(n*k**3, 3)         # (n*k^3, 3)
        filter = 0
        for i in range(8):
            indice_xyz = Variable(xyz.data.floor() + delta[i:i+1])   # (n*k^3, 3)
            valid_xyz = (indice_xyz >= 0) * (indice_xyz <= k-1)
            indice_x = indice_xyz[:, 0]
            indice_y = indice_xyz[:, 1]
            indice_z = indice_xyz[:, 2]
            c_filter = weight_3d[batch_indice, torch.clamp(indice_x,0,k-1).long(), torch.clamp(indice_y,0,k-1).long(), torch.clamp(indice_z,0,k-1).long()]   # (n*k^3,)
            c_filter *= (1 - torch.abs(xyz[:, 0]-indice_x)) * (1 - torch.abs(xyz[:, 1]-indice_y)) * (1 - torch.abs(xyz[:, 2]-indice_z))
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
        epsilon = 0.000000001   # used to handle 0/0
        v_length = torch.sqrt(torch.sum(v*v, dim=1))
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

        I3 = Variable(torch.eye(3).view(1,3,3).cuda())
        R = I3 + torch.sin(theta).view(n,1,1)*m + (1-torch.cos(theta)).view(n,1,1)*torch.bmm(m, m)
        return R

    def _get_3d_indice(self, k):
        """
        indice: (k^3, 3)
        """
        a = torch.arange(0,k**3).long()
        indice = torch.stack([a/(k*k), a%(k*k)/k, a%(k)], dim=1).float()          # (k^3, 3)
        return indice

class FastRotation(nn.Module):
    def __init__(self, num_channels, kernel_size, padding_mode='zeros'):
        """
        theta_v: (n, 3)
        theta: (n,)
        """
        super(FastRotation, self).__init__()
        self.n = num_channels
        self.k = kernel_size
        self.padding_mode = padding_mode
        
        self.base_grids = self._get_base_grids(self.n, self.k)

    def forward(self, input_filter, theta_v, theta, extra_alpha=None):
        """
        input_filter: (**, k, k, k)
        output_filter: same as input_filter
        """
        output_filter = input_filter.view(-1,self.k,self.k,self.k)
        assert output_filter.size(0) == self.n

        output_filter = self._linear_interpolation(
            theta_v, theta, output_filter, padding_mode=self.padding_mode, extra_alpha=extra_alpha
        )
        output_filter = output_filter.view(input_filter.size())
        return output_filter

    def _linear_interpolation(self, theta_v, theta, weight_3d, padding_mode, extra_alpha=None):
        k = self.k

        weight_3d = weight_3d.view(-1, 1, k, k, k)

        base_grid = self.base_grids.view(-1, k * k * k, 3)

        R = self._get_R_by_expm(theta_v, theta)
        if not extra_alpha is None:
            extra_R = self._get_extra_R(extra_alpha).unsqueeze(0)
            # R = torch.bmm(R, extra_R.expand_as(R))
            R = torch.bmm(extra_R.expand_as(R), R)
        grids = torch.bmm(base_grid.view(-1, k * k * k, 3), R)
        grids = grids.view(-1, k, k, k, 3)

        outputs = F.grid_sample(weight_3d, grids, padding_mode=padding_mode)
        return outputs

    def _get_R_by_expm(self, theta_v, theta):
        """
        theta_v: (o*i, 3)
        theta: (o*i,)
        return:
            R: (o*i, 3, 3)    
        """
        n = theta_v.size(0)

        theta_v = F.normalize(theta_v, p = 2)
        vx = theta_v[:, 0]
        vy = theta_v[:, 1]
        vz = theta_v[:, 2]

        m = Variable(torch.zeros(n, 3, 3)).cuda()
        m[:, 0, 1] = -vz
        m[:, 0, 2] = vy
        m[:, 1, 0] = vz
        m[:, 1, 2] = -vx
        m[:, 2, 0] = -vy
        m[:, 2, 1] = vx

        I3 = Variable(torch.eye(3).view(1,3,3).cuda())
        R = I3 + torch.sin(theta).view(n,1,1)*m + (1-torch.cos(theta)).view(n,1,1)*torch.bmm(m, m)
        return R

    def _get_base_grids(self, n, k):
        """
        return (n, k, k, k, 3)
        """
        base_grids = to_var(torch.zeros(n, k, k, k, 3))
        for kk in range(k):
            base_grids[:, kk, :, :, 0] = kk * 2. / (k - 1) - 1
            base_grids[:, :, kk, :, 1] = kk * 2. / (k - 1) - 1
            base_grids[:, :, :, kk, 2] = kk * 2. / (k - 1) - 1
        return base_grids
    
    def _get_extra_R(self, extra_alpha):
        """
        extra_R: (3,3)
        """
        extra_R = to_var(torch.Tensor([
            [np.cos(extra_alpha),-np.sin(extra_alpha),0],
            [np.sin(extra_alpha),np.cos(extra_alpha),0],
            [0,0,1]
        ]))
        # extra_R = extra_R.unsqueeze(0).repeat(n,1,1)
        return extra_R




