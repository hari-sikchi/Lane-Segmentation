import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data
from datasets.augmentations import *

class kittiLoader(data.Dataset):
    def __init__(self, root, split="train", 
                 is_transform=False, img_size=None, augmentations=None, img_norm=True):
        self.root = root
        self.split = split
        self.img_size = [375, 1242]
        self.augmentations = augmentations
        self.mean = np.array([72.39, 82.91, 73.16])
        self.n_classes = 2

        if split == "train":
            self.files = os.listdir(root+'training/image_2/')[:230]
        elif split == "val":
            self.files = os.listdir(root+'training/image_2/')[230:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = self.files[index]
        img_path = self.root + '/training/image_2/' + img_name
        lbl_path = self.root + '/training/gt_image_2/' + img_name.split('_')[0]+'_road_'+img_name.split('_')[1]

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)
        
        #most images had size (375,1242), but some had size (376, 1241)
        img = m.imresize(img, self.img_size)
        lbl = m.imresize(lbl, self.img_size)

        if self.augmentations is not None and self.split == "train":
            img, lbl = self.augmentations(img, lbl)
        
        img = img[:, :, [1, 0, 2]]
        img = img.astype(np.float64)
        img -= self.mean
        img = img.transpose(2, 0, 1)

        road = np.zeros(self.img_size)
        road[lbl[:,:,2] == 255] = 1

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(road).long()

        return img, lbl

if __name__ == '__main__':
    local_path = '/home/deeplearning/work/Tejus/data_road/'
    augmentations = Compose([RandomRotate(10),
                             RandomHorizontallyFlip()])
    
    dst = kittiLoader(local_path, split="val", augmentations=augmentations)
    print(dst.files)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        print(imgs.shape, labels.shape)
