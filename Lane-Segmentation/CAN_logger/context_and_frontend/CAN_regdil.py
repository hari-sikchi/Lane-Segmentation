import torch
import torch.nn as nn
from torch.autograd import Variable
from datasets.tusimple.config import CONFIG

class CAN(nn.Module):

    def __init__(self):
        super(CAN, self).__init__()
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
            nn.ReLU(inplace=True),
            
            nn.ZeroPad2d(1),
           
            # context module

            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=(0,0), bias=True, dilation=1), #ctx_conv
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=(0,0), bias=True, dilation=1),
            nn.ReLU(inplace=True),
            
            nn.ZeroPad2d(2),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=(0,0), bias=True, dilation=2),
            nn.ReLU(inplace=True),
           
            nn.ZeroPad2d(4),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=(0,0), bias=True, dilation=4),
            nn.ReLU(inplace=True),
            
            nn.ZeroPad2d(8),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=(0,0), bias=True, dilation=8),
            nn.ReLU(inplace=True),
            
            nn.ZeroPad2d(16),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=(0,0), bias=True, dilation=16),
            nn.ReLU(inplace=True),

            nn.ZeroPad2d(1),
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=0, bias=True, dilation=1),
            nn.ReLU(inplace=True),

            #nn.ZeroPad2d(1),
            nn.Conv2d(2, 2, kernel_size=1, stride=1, padding=0, bias=False, dilation=1), 
            # nn.ZeroPad2d(32),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True, dilation=32),
            # nn.ReLU(inplace=True),

            # nn.ZeroPad2d(64),
            # nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=1, bias=True, dilation=64),
            # nn.ReLU(inplace=True),

            # nn.ZeroPad2d(1),
            # nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.LeakyReLU(inplace=True),
            # nn.Conv2d(19, 2, kernel_size=1, stride=1, padding=0, bias=True),
            
            nn.Upsample(size=(CONFIG['tusimple']['output_shape'][0],CONFIG['tusimple']['output_shape'][1]), mode='bilinear'),
            
            # nn.Conv2d(19, 19, kernel_size=16, stride=1, padding=7, bias=False),
            # nn.ReLU(inplace=True),
            
            #nn.Softmax(dim=1)            
    )

    def forward(self, x):
        x = self.features(x)
        return x

def activ_forward_hook(self, inputs, outputs):
    #print(len(inputs), len(outputs))
    #print(torch.sum(inputs!=outputs))
    print(inputs[0].shape, outputs[0].shape)
    print(torch.sum(inputs[0][:,:,1:67,1:137] != outputs[0].unsqueeze(0)))
    print(inputs[0][:,:,1:67,1:137])
    print(outputs[0].unsqueeze(0))
    
    print("-------------------")

# net = CAN()
# x = Variable(torch.randn(1,3,720,1280))
# for m in net.modules():
#     if isinstance(m, nn.Conv2d):
#         print(m)
#         m.register_forward_hook(activ_forward_hook)
# o = net(x)

# net = CAN()
# net.load_state_dict(torch.load('test_identity.wts'))
# print(net.state_dict()['features.36.weight'])
# print(net.state_dict()['features.36.bias'])
# x = Variable(torch.randn(1,3,720,1280))
# x.fill_(1)
# i = 0
# for m in net.modules():
#     i += 1
#     if i<= 36:
#         continue
#     if i>=55:
#         break
#     if isinstance(m, nn.Conv2d):
#         print(m)
#         m.register_forward_hook(activ_forward_hook)
# o = net(x)

   
