import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import os, scipy.io
from scipy import misc
from skimage import io
from datasets.config import CONFIG
from datasets.kitti_loader import kittiLoader
from FCN_Dilated import FCN_Dilated
from loss import cross_entropy2d
from datasets.augmentations import *
from torch.utils import data
from metrics import runningScore

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

net = FCN_Dilated()
net.load_state_dict(torch.load('vgg_fcn_pretrained.wts', map_location=lambda storage, location: storage))
net.to(device)

print(net.features[33].weight.shape)

w = net.features[33].weight

print(torch.min(w[0,:,:,:]))
print(torch.max(w[0,:,:,:]))

print(torch.min(w[1,:,:,:]))
print(torch.max(w[1,:,:,:]))


