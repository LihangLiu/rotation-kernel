import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learnx.torch import *
from learnx.torch.nn import *
from learnx.torch.nn.functional import *


class ConvRotate3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_mode = '3d', kernel_rotate = True,
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
            dimensions = [3]
        elif self.kernel_mode == '2d+1d':
            dimensions = [2, 1]
        elif self.kernel_mode == '1d+1d+1d':
            dimensions = [1, 1, 1]
        else:
            raise NotImplementedError('unsupported kernel mode: {0}'.format(self.kernel_mode))

        self.kernels, self.masks = nn.ParameterList(), nn.ParameterList()
        for d in dimensions:
            size = (self.out_channels, self.in_channels) + (self.kernel_size,) * d
            self.kernels.append(nn.Parameter(torch.zeros(*size)))
            self.masks.append(nn.Parameter(torch.zeros(*size)))

        for kernel, mask in zip(self.kernels, self.masks):
            nn.init.normal(kernel, std = 0.02)
            nn.init.constant(mask, val = 1)

        if self.kernel_rotate:
            self.theta_n = nn.Parameter(torch.zeros(self.out_channels * self.in_channels, 3))
            nn.init.uniform(self.theta_n, a = 0, b = 1)

            self.theta_r = nn.Parameter(torch.zeros(self.out_channels * self.in_channels))
            nn.init.uniform(self.theta_r, a = 0, b = np.pi)

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.bias = None

    def forward(self, inputs):
        i, o, k = self.in_channels, self.out_channels, self.kernel_size

        kernels = as_variable(torch.ones(o * i))
        for kernel, mask in zip(self.kernels, self.masks):
            kernels = torch.bmm(
                kernels.view(o * i, -1, 1),
                (kernel * mask.detach()).view(o * i, 1, -1)
            )
        kernels = kernels.view(o, i, k, k, k)

        if self.kernel_rotate:
            grids = rotate_grid(theta_n = self.theta_n, theta_r = self.theta_r, size = (o * i, 1, k, k, k))

            kernels = kernels.view(o * i, 1, k, k, k)
            kernels = F.grid_sample(kernels, grids)
            kernels = kernels.view(o, i, k, k, k)

        outputs = F.conv3d(inputs, kernels, bias = self.bias, stride = self.stride, padding = self.padding)
        return outputs

    def prune(self):
        for k, (kernel, mask) in enumerate(zip(self.kernels, self.masks)):
            kernel = np.abs(as_numpy(kernel))
            values = np.sort(kernel.reshape(-1))
            threshold = values[len(values) // 2]
            print(threshold)

            new_mask = (kernel >= threshold).astype(float)
            print(np.sum(new_mask), np.prod(new_mask.shape))
            self.masks[k].data = torch.from_numpy(new_mask).float().cuda()


class ConvNet3d(nn.Module):
    def __init__(self, channels, features, kernel_mode, kernel_rotate, normalization = 'batch'):
        super(ConvNet3d, self).__init__()

        layers = []
        for k in range(len(channels) - 1):
            in_channels = channels[k]
            out_channels = channels[k + 1]

            layers.append(ConvRotate3d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 4,
                kernel_mode = kernel_mode,
                kernel_rotate = kernel_rotate,
                stride = 1,
                padding = 1,
                bias = False
            ))

            if normalization is not None and normalization is not False:
                layers.append(get_normalization(layer_type = normalization, num_dims = 3)(out_channels))

            layers.append(nn.LeakyReLU(
                negative_slope = 0.2,
                inplace = True
            ))
            layers.append(nn.MaxPool3d(
                kernel_size = 3,
                stride = 2,
                padding = 1
            ))

        self.extractor = nn.Sequential(*layers)

        self.classifier = LinearLayers(
            features = [channels[-1]] + features,
            normalization = normalization
        )
        self.apply(init_weights())

    def forward(self, inputs):
        features = self.extractor.forward(inputs).view(inputs.size(0), -1)
        outputs = self.classifier.forward(features)
        return outputs
