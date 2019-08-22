import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from datasets.config import CONFIG
from datasets.tusimple_loader import tusimpleLoader
from CAN import CAN
from datasets.augmentations import *
import visdom
import torchvision

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

net = CAN()
pretrained_state_dict = torch.load('models/CAN_pretrained.wts')
pretrained_state_dict = {k:v for k, v in pretrained_state_dict.items() if 'features.62' not in k and 'features.64' not in k}
net.load_state_dict(pretrained_state_dict, strict=False)
net.to(device)

print(device)
net.segment[0].weight.to(device)
net.segment[0].bias.to(device)
print(device)
print(net.features[0].weight.device)
print(net.segment[0].weight.device)

augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip()])
dataset = tusimpleLoader('/home/tejus/Downloads/train_set/', split="train", augmentations=augmentations)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = 0.00001, momentum=0.9) # 0.00001
# loss_fn = nn.BCEWithLogitsLoss()
loss_fn = nn.CrossEntropyLoss()

def update_lr(optimizer, epoch):
    if epoch == 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    if epoch == 30:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

net.train()
for EPOCHS in range(40):
    update_lr(optimizer, EPOCHS)
    running_loss = 0
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        out = net(imgs)

        # loss = loss_fn(out, labels.unsqueeze(1).float())
        loss = loss_fn(out, labels)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        if i%300 == 299:
            print("EPOCH %d: %d/3000 loss=%f"%(EPOCHS, i+1, running_loss/300))
            running_loss = 0

    #torch.save(net.state_dict(), 'trained.wts')
