import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional

from learnx.torch import *
from learnx.torch.nn import *
from learnx.torch.nn.functional import *

__all__ = ['ConvRotateNet3d']


class ConvRotate3d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, kernel_mode = None, kernel_rotate = True,
                 stride = 1, padding = 0, bias = True):
        super(ConvRotate3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_mode = kernel_mode
        self.kernel_rotate = kernel_rotate
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.kernel_mode == '3d':
            dims = [3]
        elif self.kernel_mode == '2d+1d':
            dims = [2, 1]
        elif self.kernel_mode == '1d+1d+1d':
            dims = [1, 1, 1]
        else:
            raise NotImplementedError('unsupported kernel mode: {0}'.format(kernel_mode))

        self.kernels, self.masks = nn.ParameterList(), nn.ParameterList()
        for dim in dims:
            size = (out_channels, in_channels) + (kernel_size,) * dim
            self.kernels.append(nn.Parameter(torch.zeros(*size)))
            self.masks.append(nn.Parameter(torch.zeros(*size)))

        for kernel, mask in zip(self.kernels, self.masks):
            nn.init.normal(kernel, std = 0.02)
            nn.init.constant(mask, val = 1)

        if self.kernel_rotate:
            self.theta_n = nn.Parameter(torch.zeros(out_channels * in_channels, 3))
            self.theta_r = nn.Parameter(torch.zeros(out_channels * in_channels))

            nn.init.uniform(self.theta_n, a = 0, b = 1)
            nn.init.uniform(self.theta_r, a = 0, b = np.pi)

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, inputs):
        i, o, k = self.in_channels, self.out_channels, self.kernel_size

        kernels = as_variable(torch.ones(o * i))
        for kernel, mask in zip(self.kernels, self.masks):
            kernels = torch.bmm(kernels.view(o * i, -1, 1), (kernel * mask.detach()).view(o * i, 1, -1))

        if self.kernel_rotate:
            kernels = kernels.view(o * i, 1, k, k, k)
            grids = rotate_grid(self.theta_n, self.theta_r, size = (o * i, 1, k, k, k))
            kernels = nn.functional.grid_sample(kernels, grids)

        kernels = kernels.view(o, i, k, k, k)
        return nn.functional.conv3d(inputs, kernels, bias = self.bias, stride = self.stride, padding = self.padding)

    # def prune(self):
    #     for k, (kernel, mask) in enumerate(zip(self.kernels, self.masks)):
    #         kernel = np.abs(as_numpy(kernel))
    #         values = np.sort(kernel.reshape(-1))
    #         threshold = values[len(values) // 2]
    #         print(threshold)
    #
    #         new_mask = (kernel >= threshold).astype(float)
    #         print(np.sum(new_mask), np.prod(new_mask.shape))
    #         self.masks[k].data = torch.from_numpy(new_mask).float().cuda()


class ConvRotateNet3d(nn.Module):
    def __init__(self, channels, features, kernel_mode, kernel_rotate):
        super(ConvRotateNet3d, self).__init__()

        num_layers = len(channels) - 1

        modules = []
        for k in range(num_layers):
            in_channels = channels[k]
            out_channels = channels[k + 1]

            modules.append(ConvRotate3d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 5,
                kernel_mode = kernel_mode,
                kernel_rotate = kernel_rotate,
                padding = 2,
                bias = False
            ))

            modules.extend([
                get_normalization('batch', out_channels, 3),
                get_nonlinear('lrelu'),
                get_subsampling('maxpool', 2, 3)
            ])

        self.extractor = nn.Sequential(*modules)
        self.classifier = LinearLayers(features = features)

        self.apply(init_weights())

    def forward(self, inputs):
        features = self.extractor.forward(inputs).view(inputs.size(0), -1)
        outputs = self.classifier.forward(features)
        return outputs
