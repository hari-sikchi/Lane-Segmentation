import sys
sys.path.append('/home/tejus/lane-seg-experiments/Segmentation/')
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from datasets.kitti.config import CONFIG
from datasets.kitti.kitti_loader import kittiLoader
from datasets.tusimple.tusimple_loader import tusimpleLoader
from CAN_frontend import CAN
from datasets.tusimple.augmentations import *
import visdom
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from metrics import runningScore 
TRAIN_BATCH = 3
VAL_BATCH = 4




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

pretrained_state_dict = models.vgg16(pretrained=True).state_dict()
pretrained_state_dict = {k:v for k, v in pretrained_state_dict.items() if 'classifier' not in k}
#rename parameters. ordering changes because maxpool layers aren't present in network.
pretrained_state_dict['features.23.weight'] = pretrained_state_dict.pop('features.24.weight')
pretrained_state_dict['features.23.bias'] = pretrained_state_dict.pop('features.24.bias')
pretrained_state_dict['features.25.weight'] = pretrained_state_dict.pop('features.26.weight')
pretrained_state_dict['features.25.bias'] = pretrained_state_dict.pop('features.26.bias')
pretrained_state_dict['features.27.weight'] = pretrained_state_dict.pop('features.28.weight')
pretrained_state_dict['features.27.bias'] = pretrained_state_dict.pop('features.28.bias')

#net.load_state_dict(torch.load('trained.wts'))

net.load_state_dict(pretrained_state_dict, strict=False)
net.to(device)
net.train()

augmentations = Compose([RandomRotate(5), RandomHorizontallyFlip()])
train_dataset = tusimpleLoader('/home/tejus/Downloads/train_set/', split="train", augmentations=augmentations)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True, num_workers=TRAIN_BATCH, pin_memory=True)

val_dataset = tusimpleLoader('/home/tejus/Downloads/train_set/', split="val", augmentations=None)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=VAL_BATCH, shuffle=True, num_workers=VAL_BATCH, pin_memory=True)
# dataset = kittiLoader('/mnt/data/tejus/kitti_road/data_road/', split="train", augmentations=augmentations)
# trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, pin_memory=True)

running_metrics_val =runningScore(2)



optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = 0.0001, momentum=0.9) # 0.00001
# loss_fn = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 2, verbose = True, min_lr = 0.000001)
loss_fn = nn.CrossEntropyLoss()

import math
best_val_loss = math.inf

for EPOCHS in range(50):
    net.train()
    running_loss = 0
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        
        out = net(imgs)
        
        loss = loss_fn(out, labels)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

        if i%100 == 99:
            print("EPOCH %d: %d/3000 loss=%f"%(EPOCHS, TRAIN_BATCH*(i+1), running_loss/(100)))
            running_loss = 0

    for param_group in optimizer.param_groups:
        print('Current learning rate is: ', param_group['lr'])
    net.eval()
    val_loss = 0
    ctr = 0
    best_iou=-100
    with torch.no_grad():
        for i, data in enumerate(valloader):
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            
            out = net(imgs)
            loss = loss_fn(out, labels)
            ctr+=1
            pred = out.data.max(1)[1]
            running_metrics_val.update(pred.cpu().numpy(),labels.cpu().numpy())
            val_loss += loss.item()
    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)

    # for k, v in class_iou.items():
    #     logger.info('{}: {}'.format(k, v))

    running_metrics_val.reset()


    print("EPOCH %d: VAL loss=%f"%(EPOCHS, val_loss/(ctr)))
    
    scheduler.step(val_loss)

    if score["Mean IoU : \t"] >= best_iou:
        best_iou = score["Mean IoU : \t"]
        torch.save(net.state_dict(), 'best_val.wts')


    # if val_loss < best_val_loss:
    #     torch.save(net.state_dict(), 'best_val.wts')

    torch.save(net.state_dict(), 'trained.wts')
