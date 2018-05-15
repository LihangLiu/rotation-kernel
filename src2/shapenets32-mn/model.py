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
    def __init__(self, nf, num_syns, kernel_mode, input_size, num_theta):
        """
        # input is (n,1,32,32,32)
        """
        super(ConvNet, self).__init__()
        self.kernel_mode = kernel_mode
        self.nf = nf
        self.input_size = input_size
        self.num_theta = num_theta

        if self.input_size == 32:
            list_channels = [1, nf, nf*2, nf*4, nf*8, nf*16]
        elif self.input_size == 64:
            list_channels = [1, nf, nf*2, nf*4, nf*8, nf*16, nf*16]

        main_layers = []
        for i in range(len(list_channels) - 1):
            if i == 0 or 'max' in self.kernel_mode:
                in_channels = list_channels[i]
            else:
                in_channels = list_channels[i] * num_theta
            out_channels = list_channels[i+1]

            main_layers.append(RotationConv3d(
                kernel_mode, in_channels, out_channels, 4, stride=1, padding=1, bias=False,
                num_theta=num_theta
            ))
            if 'mn' in self.kernel_mode and not 'max' in self.kernel_mode:
                main_layers.append(nn.BatchNorm3d(out_channels * num_theta))
            else:
                main_layers.append(nn.BatchNorm3d(out_channels))
            main_layers.append(nn.LeakyReLU(0.2, inplace=True))
            main_layers.append(nn.MaxPool3d(3, stride=2, padding=1))

        self.main = nn.Sequential(*main_layers)
        if 'mn' in self.kernel_mode and not 'max' in self.kernel_mode:
            self.last_channels = list_channels[-1] * num_theta
        else:
            self.last_channels = list_channels[-1]
        self.fcs = nn.Sequential(
            nn.Linear(self.last_channels, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_syns),
        )


    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self.last_channels)
        output = self.fcs(output)
        return output

    def weight_aggregate_2_share(self):
        for m in self.main:
            if isinstance(m, RotationConv3d):
                m.weight_aggregate_2_share()

    def choose_weight(self, number):
        for m in self.main:
            if isinstance(m, RotationConv3d):
                m.choose_weight(number)



if __name__ == '__main__':

    net = ConvNet(nf=8, num_syns=40, kernel_mode='3d_rot')
    net.cuda()
    # net.apply(weights_init)
    # print(net)

    # print('- - - state_dict - - -')
    # print(net.state_dict().keys())

    # criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    input = Variable(torch.randn(2, 1, 32, 32, 32)).cuda()
    labels = Variable(torch.ones(2)).long().cuda()
    for epoch in range(10):
        # optimizer.zero_grad()
        out = net(input)
        loss = criterion(out, labels)
        loss.backward()
        print(loss.data[0])
        optimizer.step()

    # path = "test.param"
    # torch.save(net.state_dict(), path)


    












