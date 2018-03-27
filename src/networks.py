from __future__ import print_function

import torch.nn as nn

from modules import ConvRotate3d
from utils.torch import weights_init


class ConvNet3D(nn.Module):
    def __init__(self, channels, kernel_mode, num_classes, batch_norm = True):
        super(ConvNet3D, self).__init__()
        self.channels = channels
        self.kernel_mode = kernel_mode
        self.num_classes = num_classes
        self.batch_norm = batch_norm

        num_layers = len(self.channels) - 1

        modules = []
        for k in range(num_layers):
            in_channels = self.channels[k]
            out_channels = self.channels[k + 1]

            modules.append(ConvRotate3d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 4,
                kernel_mode = self.kernel_mode,
                stride = 1,
                padding = 1,
                bias = False
            ))

            if self.batch_norm:
                modules.append(nn.BatchNorm3d(out_channels))

            modules.append(nn.LeakyReLU(0.2, True))
            modules.append(nn.MaxPool3d(3, stride = 2, padding = 1))

        self.network = nn.Sequential(*modules)

        self.classifier = nn.Sequential(
            nn.Dropout3d(.5),
            nn.Linear(self.channels[-1], 128),
            nn.Dropout3d(.5),
            nn.Linear(128, num_classes),
        )
        self.apply(weights_init)

    def forward(self, inputs):
        outputs = self.network.forward(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.classifier.forward(outputs)
        return outputs
