import sys
sys.path.append('/home/tejus/lane-seg-experiments/Segmentation/')

import torch
import torchvision
from torchvision import datasets, models, transforms
from datasets.tusimple.config import CONFIG
from datasets.kitti.augmentations import *
from datasets.kitti.kitti_loader import kittiLoader
from datasets.tusimple.tusimple_loader import tusimpleLoader
from CAN import CAN
import matplotlib.pyplot as plt
from metrics import runningScore 

use_device = "cpu"

device = torch.device(use_device)

dataset = tusimpleLoader('/mnt/data/tejus/train_set/', split="val")
checkpoint_dir = 'best_val_model.pkl'

if use_device == "cuda:0":
    valloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    checkpoint = torch.load(checkpoint_dir)

else:
    valloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    checkpoint = torch.load(checkpoint_dir, map_location=lambda storage, loc: storage)


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

    plt.imsave(img_path, np.transpose(npimg, (1, 2, 0)), format="png", cmap="hot")

net = CAN()

net.load_state_dict(checkpoint["model_state"])
net.to(device)
net.eval()

i = 0
save_data = False
running_metrics_val = runningScore(2)

with torch.no_grad():
    for data in valloader:
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        
        out = net(imgs)

        if save_data:
            save_img(imgs, 'images/'+str(i)+'.png', True)
        
        out = torch.nn.functional.softmax(out, 1)[:,1,:,:]
        
        out = torch.where(out.cpu() > 0.35, torch.Tensor([1]), torch.Tensor([0]))
        
        running_metrics_val.update(out.cpu().numpy(),labels.cpu().numpy())
        if save_data:
            save_img(out, 'images/out'+str(i)+'.png', False)
        
        i += 1
        print(i)
        if i>20:
            break

    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)
    print(class_iou)
    # for k, v in class_iou.items():
    #     print(k, v)

    running_metrics_val.reset()
