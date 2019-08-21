import sys
sys.path.append('/home/tejus/lane-seg-experiments/Segmentation/')

import torch
import torchvision
from torchvision import datasets, models, transforms
from datasets.tusimple.config import CONFIG
from datasets.kitti.augmentations import *
from datasets.kitti.kitti_loader import kittiLoader
from datasets.tusimple.tusimple_loader import tusimpleLoader
from CAN_lfe import CAN
import matplotlib.pyplot as plt

device = torch.device("cpu")

# dataset = kittiLoader('/mnt/data/tejus/kitti_road/data_road/', split="train")
dataset = tusimpleLoader('/mnt/data/tejus/train_set/', split="val")
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

import matplotlib.pyplot as plt
def save_img(img, img_path, is_image):
    if is_image:
        img[:,0,:,:] *= 0.229
        img[:,1,:,:] *= 0.224
        img[:,2,:,:] *= 0.225

        img[:,0,:,:] += 0.485
        img[:,1,:,:] += 0.456
        img[:,2,:,:] += 0.406

        img = img[:,[1,0,2],:,:]
    img = torchvision.utils.make_grid(img)
    npimg = img.cpu().numpy()
    # npimg[npimg > 0.1] = 1
    # npimg[npimg <= 0.1] = 0
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(img_path)

net = CAN()
net.load_state_dict(torch.load('best_val.wts'))
net.to(device)
net.eval()

i = 0
with torch.no_grad():
    for data in trainloader:
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        save_img(imgs, 'images/'+str(i)+'.png', True)
        
        out = net(imgs)
        
        # For BCE Loss
        # out = torch.nn.functional.sigmoid(out)
        # out = torch.where(out > 0.2, torch.Tensor([1]).cuda(), torch.Tensor([0]).cuda())
        # out = out.data.max(1)[1]
        # plt.imshow(out[0][1])
        # plt.savefig('images/out' + str(i) + '.png')

        out = torch.nn.functional.softmax(out, 1)[:,1,:,:]
        out = out.data.max(0)[1]
        # out = torch.where(out > 0.5, torch.Tensor([1]), torch.Tensor([0]))

        save_img(out, 'images/out'+str(i)+'.png', False)
        #visualize using pyplot
        
        # plt.imshow(out[0])
        # plt.savefig('images/out' + str(i) + '.png')
          
        

        i += 1
        if i>10:
            break
