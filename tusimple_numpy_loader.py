import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import cv2

from torch.utils import data
# from datasets.augmentations import *

class tusimpleLoader(data.Dataset):
    def __init__(self, root, split="train", 
                 is_transform=False, img_size=None, augmentations=None, img_norm=True,channel_split=False):
        """
        channel_split : whether channels must be seperated for the classes
        """
        self.root = root
        self.split = split
        self.img_size = [720, 1280]
        self.augmentations = augmentations
        # self.mean = np.array([72.39, 82.91, 73.16])
        self.mean = np.array([0.0, 0.0, 0.0])
        self.n_classes = 2
        self.channel_split = channel_split
        if split == "train":
            self.images = np.load(root + 'npy/tu-simple/X_train.npy')
            self.labels = np.load(root + 'npy/tu-simple/Y_train.npy')
        else:
            self.images = np.load(root + 'npy/tu-simple/X_test.npy')
            self.labels = np.load(root + 'npy/tu-simple/Y_test.npy')
        
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        img = self.images[index]
        # print(img.shape)
        img = np.reshape(img,[3,*self.img_size])
        lbl = self.labels[index]
        # print(img.shape)
        # img = img.astype(np.float64)
        # img = imgs[0].cpu().numpy()
        # print(img.shape)
        # print(np.max(img))
        # # img = img*255
        # img = img.astype(np.uint8)
        # img = img.reshape(720, 1280, 3)
        # print('transposed', img.shape)

        # cv2.imshow('a', img)
        # cv2.waitKey(0)
        # # print(img.shape)
        # print(self.mean.shape)
        # self.mean = self.mean.reshape(3,1,1)
        # img -= self.mean

        # print(self.img_size)
        lbl = lbl.astype(np.float64)
        lbl = np.reshape(lbl,[1,*self.img_size])
        # print(lbl.shape)
        # print(img.shape)

        label = np.zeros((2,*self.img_size)).astype(np.float64)
        # print(label.shape)
        # print(np.where(lbl == 255))
        label[1,np.where(lbl > 125)[1],np.where(lbl > 125)[2]] = 1.0
        label[0,np.where(lbl < 100)[1],np.where(lbl < 100)[2]] = 1.0
        # print("LABEL INFORMATION")
        # print(np.min(label))
        # print(np.max(label))
        # cv2.imshow("LABEL 0",label[0].reshape(720,1280,1))
        # cv2.imshow("LABEL 1",label[1].reshape(720,1280,1))
        # cv2.waitKey   (0)

        return img, label

# if __name__ == '__main__':
#     local_path = '/home/tejus/Downloads/train_set/'
#     augmentations = Compose([RandomRotate(10),
#                              RandomHorizontallyFlip()])
#     print("hello")
# dst = tusimpleLoader(local_path, split="val", augmentations=augmentations)

# print(dst.files)
# trainloader = data.DataLoader(dst, batch_size=4)

# for i, data in enumerate(trainloader):
#     imgs, labels = data
#     print(imgs.shape, labels.shape)
#     break