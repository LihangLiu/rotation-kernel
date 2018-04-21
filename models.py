import functools

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import conv3d, grid_sample

import learnx.torch as tx


class ConvRotate3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_mode = None, kernel_rotate = True,
                 stride = 1, padding = 0, dilation = 1, groups = 1, bias = True):
        super(ConvRotate3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_mode = kernel_mode
        self.kernel_rotate = kernel_rotate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        if kernel_mode == '3d':
            dimensions = [3]
        elif kernel_mode == '2d+1d':
            dimensions = [2, 1]
        elif kernel_mode == '1d+1d+1d':
            dimensions = [1, 1, 1]
        else:
            raise NotImplementedError('unsupported kernel mode: {0}'.format(kernel_mode))

        self.kernels, self.masks = nn.ParameterList(), nn.ParameterList()
        for dimension in dimensions:
            size = (out_channels, in_channels) + (kernel_size,) * dimension
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

        kernels = tx.as_variable(torch.ones(o * i))
        for kernel, mask in zip(self.kernels, self.masks):
            kernels = torch.bmm(kernels.view(o * i, -1, 1), (kernel * mask.detach()).view(o * i, 1, -1))

        if self.kernel_rotate:
            kernels = grid_sample(
                input = kernels.view(o * i, 1, k, k, k),
                grid = tx.rotate_grid(
                    theta_n = self.theta_n,
                    theta_r = self.theta_r,
                    size = (o * i, 1, k, k, k)
                )
            )

        outputs = conv3d(
            input = inputs,
            weight = kernels.view(o, i, k, k, k),
            bias = self.bias,
            stride = self.stride,
            padding = self.padding,
            dilation = self.dilation,
            groups = self.groups
        )
        return outputs

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

        backend = functools.partial(
            ConvRotate3d,
            kernel_mode = kernel_mode,
            kernel_rotate = kernel_rotate
        )

        self.extractor = tx.layers.Conv3d(
            channels = channels,
            kernel_sizes = 5,
            strides = 2,
            backend = backend,
            normalization = nn.BatchNorm3d,
            activation = nn.LeakyReLU,
            subsampling = nn.AvgPool3d,
            last_activation = True
        )

        self.classifier = tx.layers.Linear(
            features = features,
            bias = True,
            normalization = nn.BatchNorm1d,
            activation = nn.LeakyReLU
        )

        tx.init_weights(self)

    def forward(self, inputs):
        features = self.extractor.forward(inputs).view(inputs.size(0), -1)
        outputs = self.classifier.forward(features)
        return outputs
