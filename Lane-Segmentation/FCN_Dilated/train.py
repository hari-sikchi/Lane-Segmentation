import sys
sys.path.append('/home/tejus/lane-seg-experiments/Segmentation/')
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
import visdom

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.utils as vutils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip()])
dataset = kittiLoader('/mnt/data/tejus/kitti_road/data_road/', split="train", augmentations=augmentations)
trainloader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
for data in trainloader:
    imgs, labels = data
    print(imgs.shape, labels.shape)
    exit(0)

def lr_decayer(optimizer, num_epoch):
    cur_epoch = num_epoch
    lr =    0.0001
    minlr = 0.00001
    decay_interval = 10
    decay_level = 0.33
    while cur_epoch >= decay_interval:
        lr = lr * decay_level
        cur_epoch -= decay_interval
        lr = max(lr, minlr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print ("New learning rate: ", lr)

net = FCN_Dilated()
net.load_state_dict(torch.load('vgg_fcn_pretrained.wts'))
net.to(device)

ctr = 0

# Freeze first 4 layers
for w in net.parameters():
    ctr += 1
    if ctr > 4:
        break
    w.requires_grad = False

show_visdom = True
if show_visdom:
    vis =visdom.Visdom()
    loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                            Y= torch.zeros((1)).cpu(), opts =dict(xlabel='minibatches',ylabel='loss',title='Training Loss',legend= ['Loss']))



augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip()])
dataset = kittiLoader('/mnt/data/tejus/kitti_road/data_road/', split="train", augmentations=augmentations)
trainloader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0002)
loss_fn = nn.CrossEntropyLoss()

virtual_batch_size = 4
best_loss = 1000.0

for EPOCHS in range(40):
    running_loss = 0
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        out = net(imgs)

        loss = loss_fn(out, labels)
        loss.backward()

        if show_visdom:
            vis.line(X=torch.ones((1,1)).cpu()*i + EPOCHS * len(trainloader),
                                    Y= torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),win = loss_window,update='append' )
        if i%virtual_batch_size == 0 or i == len(dataset)-1:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        if i%50 == 0:
            print(str(i)+'/230')

    print("Epoch %d: Loss = %f Best Loss = %f" % (EPOCHS, running_loss/230.0, best_loss))
    
    lr_decayer(optimizer, EPOCHS)
    torch.save(net.state_dict(), 'trained.wts')
    
    pred = out.data.max(1)[1].cpu().numpy()[0]
    plt.imshow(pred)
    plt.savefig('output/' + str(EPOCHS) + '_out.png')

    vutils.save_image(imgs.squeeze(0).cpu().data, 'output/' + str(EPOCHS) + '_orig.png', normalize=True)

    if running_loss < best_loss:
        best_loss = running_loss
        torch.save(net.state_dict(), 'best_loss.wts')
        

torch.save(net.state_dict(), 'trained_15-06.wts')
