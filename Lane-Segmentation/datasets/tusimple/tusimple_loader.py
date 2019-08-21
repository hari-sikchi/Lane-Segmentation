import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from skimage import io
from torch.utils import data
import sys

sys.path.append("/mnt/data/tejus/Lane-Segmentation/datasets/tusimple")
sys.path.append("/mnt/data/tejus/Lane-Segmentation/CAN_LFE")

from augmentations import *
# from augmentations import *

json_file = open("/mnt/data/tejus/test/final_test_json.json").read().split('\n')


class tusimpleLoader(data.Dataset):
    def __init__(self, root, split="train", 
                 is_transform=False, img_size=None, augmentations=None, img_norm=True):
        self.root = root
        self.split = split
        self.img_size = [720, 1280]
        self.augmentations = augmentations
        self.mean = np.array([72.39, 82.91, 73.16])
        self.n_classes = 2
        
        if split == "train":
            self.files = os.listdir(root + 'train_data/img')[0:3000]
        elif split == "val":
            self.files = os.listdir(root + 'train_data/img')[3000:]
        elif split == "test":
            self.files = os.listdir(root + '../test/clips/all/')
        #print(self.files[:10])
            

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        if(self.split == "test"):
            img_name = self.files[index]
            img_path = self.root + '../test/clips/all/' + img_name
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

            num = img_name[: len(img_name) - 4]
            num = int(num)
            json_data = json_file[num]
            return img, json_data




        img_name = self.files[index]
        img_path =  self.root + 'train_data/img/' + img_name
        lbl_path = self.root + 'train_data/labels/' + img_name
        # print("index = ", index)
        print(img_name)
        img = io.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = io.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)
        
        img = m.imresize(img, self.img_size)
        lbl = m.imresize(lbl, self.img_size)

        if self.augmentations is not None and self.split == "train":
            img, lbl = self.augmentations(img, lbl)
        
        img = img[:, :, [1, 0, 2]]
        img = img.astype(np.float64)
        img /= 255
        img -= np.array([0.485, 0.456, 0.406])
        img /= np.array([0.229, 0.224, 0.225])
        img = img.transpose(2, 0, 1)

        road = np.zeros(self.img_size)
        road[lbl[:,:] > 128] = 1

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(road).long()

        return img, lbl

if __name__ == '__main__':
    local_path = '/mnt/data/tejus/train_set/'
    # augmentations = Compose([RandomRotate(10),
    #                          RandomHorizontallyFlip()])
    
    dst = tusimpleLoader(local_path, split="test")
    
    # print(dst.files)
    trainloader = data.DataLoader(dst, batch_size=1)
    
    for i, data in enumerate(trainloader):
        imgs, d = data
        print(d)
        if(i == 3):
            break
