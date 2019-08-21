import torch
import torch.nn as nn
from datasets.config import CONFIG

class FCN_Dilated(nn.Module):

    def __init__(self):
        super(FCN_Dilated, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=3, bias=True, dilation=4), #fc6 layer
            nn.ReLU(inplace=True),
            
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0, bias=True), #fc7 layer
            nn.ReLU(inplace=True),
           
            nn.Conv2d(4096, 2, kernel_size=1, stride=1, padding=0, bias=True), #final layer
            nn.LeakyReLU(inplace=True),
           
            nn.Upsample(size=(CONFIG['kitti']['output_shape'][0],CONFIG['kitti']['output_shape'][1]), mode='bilinear'),
            
    )

    def forward(self, x):
        x = self.features(x)
        return x
   
