from __future__ import print_function

import torch.nn as nn

from modules import ConvRotate3d
from utils.torch import weights_init, DensePool


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

        self.extractor = nn.Sequential(*layers)
        self.classifier = DensePool(
            features = [self.channels[-1]] + [128] + [self.num_classes],
            batch_norm = self.batch_norm,
            # dropout = self.dropout
        )
        self.apply(weights_init)

    def forward(self, inputs):
        features = self.extractor.forward(inputs).view(inputs.size(0), -1)
        outputs = self.classifier.forward(features)
        return outputs