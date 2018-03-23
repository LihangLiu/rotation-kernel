from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from modules.rotationconv3d import RotationConv3d

# custom weights initialization called on netG and netD
def weights_init(m):
    if isinstance(m, nn.Conv3d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, RotationConv3d):
        m.weight.data.normal_(0.0, 0.02)
        m.weight_1d.data.fill_(1.0)


class ConvNet(nn.Module):
    def __init__(self, nf=8, num_syns=40):
        """
        # input is (n,1,32,32,32)
        """
        super(ConvNet, self).__init__()
        self.nf = nf
        self.main = nn.Sequential(
            RotationConv3d(1, nf, 4, stride=1, padding=1, bias=False),   # (n,nf,32,32,32)
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1), 

            RotationConv3d(nf, nf*2, 4, stride=1, padding=1, bias=False),   # (n,nf*2,16,16,16)
            nn.BatchNorm3d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1), 

            RotationConv3d(nf*2, nf*4, 4, stride=1, padding=1, bias=False),   # (n,nf*4,8,8,8)
            nn.BatchNorm3d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1), 

            RotationConv3d(nf*4, nf*8, 4, stride=1, padding=1, bias=False),   # (n,nf*8,4,4,4)
            nn.BatchNorm3d(nf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1), 

            RotationConv3d(nf*8, nf*16, 4, stride=1, padding=1, bias=False),   # (n,nf*16,2,2,2)
            nn.BatchNorm3d(nf*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1), 
        )
        self.fcs = nn.Sequential(
            nn.Dropout3d(p=0.5),
            nn.Linear(1*1*1*nf*16, 128),
            nn.Dropout3d(p=0.5),
            nn.Linear(128, num_syns),
        )

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 1*1*1*self.nf*16)
        output = self.fcs(output)
        return output



if __name__ == '__main__':
    net = ConvNet(nf=32, num_syns=40)
    net.cuda()
    net.apply(weights_init)
    print(net)

    print('- - - state_dict - - -')
    print(net.state_dict().keys())

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    input = Variable(torch.randn(32, 1, 32, 32, 32)).cuda()
    labels = Variable(torch.ones(32)).long().cuda()
    for epoch in range(10):
        optimizer.zero_grad()
        out = net(input)
        loss = criterion(out, labels)
        loss.backward()
        print(loss.data[0])
        optimizer.step()

    # path = "test.param"
    # torch.save(net.state_dict(), path)


    












