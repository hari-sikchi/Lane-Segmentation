import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from skimage import io
from torch.utils import data
from datasets.tusimple.augmentations import *
# from augmentations import *



class indianLoader(data.Dataset):
    def __init__(self, root, split="test", 
                 is_transform=False, img_size=None, augmentations=None, img_norm=True):
        self.root = root
        self.split = split
        self.img_size = [720, 1280]
        self.augmentations = augmentations
        self.mean = np.array([146.16555256, 136.39416436, 106.53344295])

        self.n_classes = 2
        self.files = os.listdir(root+'indian/')


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = self.files[index]
        img_path = self.root + '/indian/' + img_name

        img = io.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        img = m.imresize(img, self.img_size)
        
        img = img[:, :, [1, 0, 2]]
        img = img.astype(np.float64)
        img /= 255
        img -= np.array([0.485, 0.456, 0.406])
        img /= np.array([0.229, 0.224, 0.225])
        img = img.transpose(2, 0, 1)


        img = torch.from_numpy(img).float()

        return img

if __name__ == '__main__':
    local_path = '/home/tejus/Downloads/train_set/'
    print(len(os.listdir(local_path +'indian/')))
    dst = indianLoader(local_path, split="test")
    
    # print(dst.files)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs = data
        print(imgs.shape)
        break