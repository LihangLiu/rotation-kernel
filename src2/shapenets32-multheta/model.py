from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import _init_paths
from utils.torchhelper import *
from utils.filehelper import *
from modules.rotationconv3d import RotationConv3d

class ConvNet(nn.Module):
    def __init__(self, nf, num_syns, kernel_mode, num_theta, input_size):
        """
        # input is (n,1,32,32,32)
        """
        super(ConvNet, self).__init__()
        self.kernel_mode = kernel_mode
        self.num_theta = num_theta
        self.nf = nf
        self.input_size = input_size
        if self.input_size == 32:
            list_channels = [1, nf, nf*2, nf*4, nf*8, nf*16]
        elif self.input_size == 64:
            list_channels = [1, nf, nf*2, nf*4, nf*8, nf*16, nf*16]

        main_layers = []
        for i in range(len(list_channels) - 1):
            in_channels = list_channels[i]
            out_channels = list_channels[i+1]
            main_layers.append(RotationConv3d(
                kernel_mode, num_theta, in_channels, out_channels, 4, stride=1, padding=1, bias=False
            ))
            main_layers.append(nn.BatchNorm3d(out_channels))
            main_layers.append(nn.LeakyReLU(0.2, inplace=True))
            main_layers.append(nn.MaxPool3d(3, stride=2, padding=1))

        self.main = nn.Sequential(*main_layers)
        self.last_channels = list_channels[-1]
        self.last_channels = list_channels[-1] * self.num_theta
        self.fcs = nn.Sequential(
            nn.Linear(self.last_channels, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, num_syns),
        )


    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self.last_channels)
        output = self.fcs(output)
        return output

    def get_spar_regular_loss(self):
        loss = 0
        for m in self.main:
            if isinstance(m, RotationConv3d):
                loss += m.get_spar_regular_loss()
        return loss

