from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvRotate3d
from utils.torch import DensePool, to_var, weights_init


class STNet(nn.Module):
    def __init__(self, channels, batch_norm = True, dropout = 0.5):
        super(STNet, self).__init__()
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
        self.classifier = DensePool(
            features = [self.channels[-1]] + [128] + [4],
            batch_norm = self.batch_norm,
            dropout = self.dropout
        )
        self.apply(weights_init)

        self.kernel_size = 32

        self.base_grids = to_var(torch.zeros(
            self.kernel_size, self.kernel_size, self.kernel_size, 3
        ))
        for k in range(self.kernel_size):
            self.base_grids[:, :, k, 0] = k * 2. / (self.kernel_size - 1) - 1
            self.base_grids[:, k, :, 1] = k * 2. / (self.kernel_size - 1) - 1
            self.base_grids[k, :, :, 2] = k * 2. / (self.kernel_size - 1) - 1

    def forward(self, inputs):
        features = self.extractor.forward(inputs).view(inputs.size(0), -1)
        outputs = self.classifier.forward(features)

        theta_n = outputs[:, :-1]
        theta_r = outputs[:, -1]

        n, k = inputs.size(0), 32

        normal = F.normalize(theta_n, p = 2)

        transform = to_var(torch.zeros(inputs.size(0), 3, 3))
        transform[:, 2, 1], transform[:, 1, 2] = normal[:, 0], -normal[:, 0]
        transform[:, 0, 2], transform[:, 2, 0] = normal[:, 1], -normal[:, 1]
        transform[:, 1, 0], transform[:, 0, 1] = normal[:, 2], -normal[:, 2]

        theta = to_var(torch.eye(3)).view(1, 3, 3) + \
                torch.sin(theta_r).view(-1, 1, 1) * transform + \
                (1 - torch.cos(theta_r)).view(-1, 1, 1) * torch.bmm(transform, transform)

        grids = self.base_grids.view(k * k * k, 3)
        grids = torch.stack([grids] * inputs.size(0), 0)
        grids = torch.bmm(grids, theta)
        grids = grids.view(-1, k, k, k, 3)

        outputs = F.grid_sample(inputs, grids)
        outputs = outputs.view(-1, 1, k, k, k)
        return outputs


class ConvNet3d(nn.Module):
    def __init__(self, channels, kernel_mode, num_classes, batch_norm = True, dropout = 0.5):
        super(ConvNet3d, self).__init__()
        self.channels = channels
        self.kernel_mode = kernel_mode
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        self.dropout = dropout

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

        self.transformer = STNet(
            channels = self.channels,
            batch_norm = self.batch_norm,
            dropout = self.dropout
        )
        self.extractor = nn.Sequential(*layers)
        self.classifier = DensePool(
            features = [self.channels[-1]] + [128] + [self.num_classes],
            batch_norm = self.batch_norm,
            dropout = self.dropout
        )
        self.apply(weights_init)

    def forward(self, inputs):
        inputs = self.transformer.forward(inputs)
        features = self.extractor.forward(inputs).view(inputs.size(0), -1)
        outputs = self.classifier.forward(features)
        return outputs
