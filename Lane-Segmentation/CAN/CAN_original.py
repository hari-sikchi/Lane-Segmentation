import torch
import torch.nn as nn
from datasets.kitti.config import CONFIG

class CAN(nn.Module):

    def __init__(self):
        super(CAN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0, bias=True, dilation=4), #fc6 layer
            nn.ReLU(inplace=True),
            
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0, bias=True), #fc7 layer
            nn.ReLU(inplace=True),
           
            nn.Conv2d(4096, 19, kernel_size=1, stride=1, padding=0, bias=True), #final layer
            nn.ReLU(inplace=True),
           
            nn.ZeroPad2d(1),
            
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True), #ctx_conv
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            
            nn.ZeroPad2d(2),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
            nn.ReLU(inplace=True),
           
            nn.ZeroPad2d(4),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True, dilation=4),
            nn.ReLU(inplace=True),
            
            nn.ZeroPad2d(8),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True, dilation=8),
            nn.ReLU(inplace=True),
            
            nn.ZeroPad2d(16),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True, dilation=16),
            nn.ReLU(inplace=True),
            
            nn.ZeroPad2d(32),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True, dilation=32),
            nn.ReLU(inplace=True),

            nn.ZeroPad2d(64),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True, dilation=64),
            nn.ReLU(inplace=True),

            nn.ZeroPad2d(1),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(19, 19, kernel_size=1, stride=1, padding=0, bias=True),
            
            nn.Upsample(size=(CONFIG['cityscapes']['output_shape'][0]+1,CONFIG['cityscapes']['output_shape'][1]+1), mode='bilinear'),
            
            nn.Conv2d(19, 19, kernel_size=16, stride=1, padding=7, bias=False),
            nn.ReLU(inplace=True),
            
            nn.Softmax(dim=1)            
    )

    def forward(self, x):
        x = self.features(x)
        return x
   
