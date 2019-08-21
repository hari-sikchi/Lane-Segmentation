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
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = FCN_Dilated()
net.load_state_dict(torch.load('best_seg_dcgan.wts'))
net.to(device)

net.eval()
dst = kittiLoader('/home-local/rohitrishabh/utilScripts/Segmentation/FCN_Dilated/data_road/', split="val")
valloader = data.DataLoader(dst, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

score = runningScore(2)

for i, data in enumerate(valloader):
    imgs, labels = data
    imgs, labels = imgs.to(device), labels.to(device)
    
    with torch.no_grad():
        out = net(imgs)

    pred = out.data.max(1)[1].cpu().numpy()
    plt.imshow(pred[0])
    plt.savefig('output_test/' + str(i) + '_out.png')
    plt.imshow(labels[0])
    plt.savefig('output_test/' + str(i) + '_orig.png')
    print(np.sum(pred))
    gt = labels.data.cpu().numpy()
    score.update(gt, pred)
    print(str(i+1)+'/59')

score, class_iou = score.get_scores()

for k, v in score.items():
    print(k, v)

for i in range(2):
    print(i, class_iou[i])

