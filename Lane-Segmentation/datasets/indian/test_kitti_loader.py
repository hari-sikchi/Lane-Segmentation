import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import os, scipy.io
from scipy import misc
from skimage import io
from datasets.config import CONFIG
from datasets.kitti_loader import kittiLoader
from loss import cross_entropy2d
from datasets.augmentations import *
from torch.utils import data


dst = kittiLoader('/home/deeplearning/work/Tejus/data_road/', split="val")
valloader = data.DataLoader(dst, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

for i, data in enumerate(valloader):
    imgs, labels = data
    labels = labels.squeeze(0)
    labels = labels.data.numpy()
    misc.imsave('sample_output{0}.png'.format(i), labels)
    
