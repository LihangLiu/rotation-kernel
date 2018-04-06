import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch import DensePool, rotate_grid, to_var, weights_init


class ConvRotate3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernel_mode, stride = 1, padding = 0, bias = True):
        super(ConvRotate3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_mode = kernel_mode
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if '3d' in self.kernel_mode:
            self.kernels = [
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size
                ))
            ]

        if '2d+1d' in self.kernel_mode:
            self.kernels = [
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
                )),
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size
                ))
            ]

        if '1d+1d+1d' in self.kernel_mode:
            self.kernels = [
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size
                )),
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size
                )),
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size
                ))
            ]

        for k, kernel in enumerate(self.kernels):
            nn.init.normal(kernel, std = 0.02)
            self.register_parameter('kernel-{0}'.format(k + 1), kernel)

        if 'rot' in self.kernel_mode:
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

        kernels = to_var(torch.ones(o * i))
        for kernel in self.kernels:
            kernels = torch.bmm(
                kernels.view(o * i, -1, 1),
                kernel.view(o * i, 1, -1)
            )
        kernels = kernels.view(o, i, k, k, k)

        if 'rot' in self.kernel_mode:
            grids = rotate_grid(theta_n = self.theta_n, theta_r = self.theta_r, size = (i * o, 1, k, k, k))

            kernels = kernels.view(o * i, 1, k, k, k)
            kernels = F.grid_sample(kernels, grids)
            kernels = kernels.view(o, i, k, k, k)

        outputs = F.conv3d(inputs, kernels, bias = self.bias, stride = self.stride, padding = self.padding)
        return outputs


# fixme
class Transformer3d(nn.Module):
    def __init__(self, channels, batch_norm = True, dropout = 0.5):
        super(Transformer3d, self).__init__()
        self.channels = channels
        self.batch_norm = batch_norm
        self.dropout = dropout

        layers = []
        for k in range(len(self.channels) - 1):
            in_channels = self.channels[k]
            out_channels = self.channels[k + 1]

            layers.append(nn.Conv3d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 4,
                stride = 1,
                padding = 1,
                bias = False
            ))

            if self.batch_norm:
                layers.append(nn.BatchNorm3d(
                    num_features = out_channels
                ))

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

        self.estimator = DensePool(
            features = [self.channels[-1], 128, 4],
            batch_norm = self.batch_norm,
            dropout = self.dropout
        )
        self.apply(weights_init)

    def forward(self, inputs):
        features = self.extractor.forward(inputs).view(inputs.size(0), -1)
        theta = self.estimator.forward(features)

        grids = rotate_grid(theta_n = theta[:, :3], theta_r = theta[:, -1], size = inputs.size())
        outputs = F.grid_sample(inputs, grids)
        return outputs


class ConvNet3d(nn.Module):
    def __init__(self, channels, kernel_mode, features, batch_norm = True, dropout = 0.5):
        super(ConvNet3d, self).__init__()
        self.channels = channels
        self.kernel_mode = kernel_mode
        self.features = features
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.transformer = Transformer3d(
            channels = self.channels,
            batch_norm = self.batch_norm,
            dropout = self.dropout
        )

        layers = []
        for k in range(len(self.channels) - 1):
            in_channels = self.channels[k]
            out_channels = self.channels[k + 1]

            layers.append(ConvRotate3d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 4,
                kernel_mode = self.kernel_mode,
                stride = 1,
                padding = 1,
                bias = False
            ))

            if self.batch_norm:
                layers.append(nn.BatchNorm3d(
                    num_features = out_channels
                ))

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

        self.classifier = DensePool(
            features = [self.channels[-1]] + self.features,
            batch_norm = self.batch_norm,
            dropout = self.dropout
        )
        self.apply(weights_init)

    def forward(self, inputs):
        # todo
        inputs = self.transformer.forward(inputs)

        features = self.extractor.forward(inputs).view(inputs.size(0), -1)
        outputs = self.classifier.forward(features)
        return outputs
