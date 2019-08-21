import sys
sys.path.append('/home/tejus/lane-seg-experiments/Segmentation/')
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from datasets.tusimple.config import CONFIG
from datasets.kitti.kitti_loader import kittiLoader
from datasets.tusimple.tusimple_loader import tusimpleLoader
from CAN_LFE.CAN_lfe import *
from datasets.tusimple.augmentations import *
# import visdom
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

def save_img(img, img_path, is_image):
    if is_image:
        img[:,0,:,:] += 72.39
        img[:,1,:,:] += 82.91
        img[:,2,:,:] += 73.16
        img /= 255
        img = img[:,[1,0,2],:,:]
    else:
        img = img.unsqueeze(1)
        img *= 255
    img = torchvision.utils.make_grid(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(img_path)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(102)

net = CAN()



net.load_state_dict(torch.load('trained.wts'), strict=False)
net.to(device)
net.train()

params = net.state_dict()
dilated_conv_layers = [36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 68, 71]
for layer_idx in dilated_conv_layers:
    w = params['features.'+str(layer_idx)+'.weight']
    b = params['features.'+str(layer_idx)+'.bias']
    w.fill_(0)
    for i in range(w.shape[0]):
        w[i,i,1,1] = 1
    #print(w)

    b.fill_(0)

    params['features.'+str(layer_idx)+'.weight'] = w 
    params['features.'+str(layer_idx)+'.weight'] = b
    
#torch.save(net.state_dict(), 'test_identity.wts')
layer_idx = 73 #56
w = params['features.'+str(layer_idx)+'.weight']
w.fill_(0)
for i in range(w.shape[0]):
    w[i,i,0,0] = 1

params['features.'+str(layer_idx)+'.weight'] = w

augmentations = Compose([RandomRotate(5), RandomHorizontallyFlip()])
train_dataset = tusimpleLoader('/home/tejus/Downloads/train_set/', split="train", augmentations=augmentations)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)

val_dataset = tusimpleLoader('/home/tejus/Downloads/train_set/', split="val", augmentations=augmentations)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)

# dataset = kittiLoader('/mnt/data/tejus/kitti_road/data_road/', split="train", augmentations=augmentations)
# trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, pin_memory=True)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = 0.0001, momentum=0.9) # 0.00001
# loss_fn = nn.BCEWithLogitsLoss()
loss_fn = nn.CrossEntropyLoss()

def update_lr(optimizer, epoch):
    if epoch == 25:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    if epoch == 40:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

import math
best_val_loss = math.inf

for EPOCHS in range(50):
    # unnecessary?
    net.train()
    update_lr(optimizer, EPOCHS)
    running_loss = 0
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        
        out = net(imgs)

        # loss = loss_fn(out, labels.float())
        loss = loss_fn(out, labels)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        if i%100 == 99:
            print("EPOCH %d: %d/1000 loss=%f"%(EPOCHS, 2*(i+1), running_loss/200))
            running_loss = 0
    
    net.eval()
    running_loss = 0
    with torch.no_grad():
        for i, data in enumerate(valloader):
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            
            out = net(imgs)
            loss = loss_fn(out, labels)
            running_loss += loss.item()
    
    print("EPOCH %d: VAL loss=%f"%(EPOCHS, running_loss/200))
    
    if running_loss < best_val_loss:
        torch.save(net.state_dict(), 'best_val_recent.wts')

    torch.save(net.state_dict(), 'trained.wts')
