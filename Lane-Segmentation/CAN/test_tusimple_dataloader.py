import sys
sys.path.append('/home/tejus/lane-seg-experiments/Segmentation/')

import torch
import torchvision
from torchvision import datasets, models, transforms
from datasets.tusimple.config import CONFIG
from datasets.tusimple.augmentations import *
from datasets.tusimple.tusimple_loader import tusimpleLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip()])
dataset = tusimpleLoader('/home/tejus/Downloads/train_set/', split="train", augmentations=augmentations)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

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

for data in trainloader:
    imgs, labels = data
    save_img(imgs, 'images/loaded_img.png', True)
    save_img(labels, 'images/loaded_label.png', False)
    break

print('image saved as images/loaded_img.png')
print('label saved as images/loaded_label.png')
    
