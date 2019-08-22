import sys
sys.path.append('/home/tejus/lane-seg-experiments/Segmentation/')

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import os, scipy.io
from scipy import misc
from skimage import io
from datasets.tusimple.config import CONFIG
from CAN_original import CAN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def segment(image):
    dataset = 'cityscapes'
    M=CONFIG[dataset]['mean_pixel']
    transform = transforms.Normalize(M, (1, 1, 1))
    image = transform(image.squeeze(0)).unsqueeze(0)

    conv_margin = CONFIG[dataset]['conv_margin']
    padding_layer = torch.nn.ReflectionPad2d(conv_margin)
    
    image.to(device)
    net.to(device)
    padding_layer.to(device)

    image = padding_layer(image)
    with torch.no_grad():
        #print(torch.min(image), torch.max(image))
        prob = net(image)
    _, out_pred = torch.max(prob, dim=1, keepdim=True)
    return out_pred

net = CAN()
net.load_state_dict(torch.load('models/CAN_pretrained.wts'))

img = np.asarray(io.imread('images/input.png'))
img = torch.from_numpy(img).float().to(device)
img = img[:, :, [1, 0, 2]].permute(2, 0, 1).unsqueeze(0) #swap color channels, 1024x2048x3-> 3x1024x2048, add batch dimension

output = segment(img)

import scipy.misc
m=CONFIG['cityscapes']['output_shape']
output=output.data.view(1,m[0],m[1]).cpu().numpy()
color_image = CONFIG['cityscapes']['palette'][output.ravel()].reshape((m[0],m[1],3))
scipy.misc.imsave('images/output.png', color_image)
