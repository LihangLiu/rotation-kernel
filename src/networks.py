import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilx.torch import as_numpy, as_variable
from utilx.torch.functional import rotate_grid
from utilx.torch.nn import LinearLayers, weights_init, Conv3dLayers


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
            self.kernels = [
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size
                ))
            ]

        if self.kernel_mode == '2d+1d':
            self.kernels = [
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
                )),
                nn.Parameter(torch.zeros(
                    self.out_channels, self.in_channels, self.kernel_size
                ))
            ]

        if self.kernel_mode == '1d+1d+1d':
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

        for kernel in self.kernels:
            nn.init.normal(kernel, std = 0.02)

        self.masks = []
        for kernel in self.kernels:
            mask = nn.Parameter(torch.ones(kernel.size()))
            self.masks.append(mask)

        for k, (kernel, mask) in enumerate(zip(self.kernels, self.masks)):
            self.register_parameter('kernel-{0}'.format(k + 1), kernel)
            self.register_parameter('mask-{0}'.format(k + 1), mask)

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

        # kernels = kernels.view(o * i, k * k, k)
        #
        # kk = []
        # for iiii in range(o * i):
        #     u, s, v = torch.svd(kernels[iiii])
        #     v = v.t().contiguous()
        #     u = u[:, 0].contiguous()
        #     s = s[0].contiguous()
        #     v = v[0, :].contiguous()
        #     r = s * torch.ger(u, v)
        #     kk.append(r)
        # kk = torch.stack(kk, dim = 0)
        # kernels = kk.view(o, i, k, k, k)

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

        self.estimator = LinearLayers(
            features = [self.channels[-1], 128, 4],
            batchnorm = self.batch_norm,
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
    def __init__(self, channels, features, kernel_mode, input_rotate, kernel_rotate, batch_norm = True, dropout = 0.5):
        super(ConvNet3d, self).__init__()
        self.channels = channels
        self.features = features
        self.kernel_mode = kernel_mode
        self.input_rotate = input_rotate
        self.kernel_rotate = kernel_rotate
        self.batch_norm = batch_norm
        self.dropout = dropout

        if self.input_rotate:
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
                kernel_rotate = self.kernel_rotate,
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

        self.classifier = LinearLayers(
            features = [self.channels[-1]] + self.features,
            batchnorm = self.batch_norm,
            dropout = self.dropout
        )
        self.apply(weights_init)

    def forward(self, inputs):
        if self.input_rotate:
            inputs = self.transformer.forward(inputs)

        features = self.extractor.forward(inputs).view(inputs.size(0), -1)
        outputs = self.classifier.forward(features)
        return outputs
