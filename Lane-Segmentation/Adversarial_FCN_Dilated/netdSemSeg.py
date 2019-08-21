
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

nc = 1
nz = 100
ngf = 64
ndf = 64


# Try InstanceNorm2d

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, 64, 3, 1, (1,0), bias=False), #1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 3, 1, (1,0), bias=False), #1
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 128, 3, 1, (1,0), bias=False), #2
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*2) x 16 x 16
            nn.Conv2d(128, 128, 3, 1, (1,0), bias=False), #3
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            # # state size. (ndf*4) x 8 x 8
            nn.Conv2d(128, 128, 3, 1, (1,0), bias=False), #4 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, (1,0), bias=False), #5
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            # # state size. (ndf*8) x 4 x 4
            nn.Conv2d(128, 256, 3, 1, (1,0), bias=False), #6
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # #7
            nn.Conv2d(256, 256, 3, 2, (1,0), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            # #8
            nn.Conv2d(256, 512, 3, 1, (1,0), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # #9
            nn.Conv2d(512, 512, 3, 1, (1,0), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            # #10
            nn.Conv2d(512, 512, 3, 1, (1,0), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # #11
            nn.Conv2d(512, 512, 3, 1, (1,0), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            # #12
            nn.Conv2d(512, 512, 3, 1, (1,0), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # #13-14c
            nn.Conv2d(512, 512, 3, 1, (1,0), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.final = nn.Sequential(
            #15
            nn.Conv2d(512, 1, 1, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)
        output = self.final(output)

        return output.view(-1, 1).squeeze(1)

