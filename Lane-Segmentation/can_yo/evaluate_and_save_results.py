import sys
sys.path.append('/home/tejus/lane-seg-experiments/Segmentation/')
import torch
import torchvision
from torchvision import datasets, models, transforms
from datasets.kitti.config import CONFIG
from datasets.kitti.augmentations import *
from datasets.kitti.kitti_loader import kittiLoader
import matplotlib.pyplot as plt
from CAN_LFE.CAN_lfe import *



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = kittiLoader('/mnt/data/tejus/kitti_road/data_road/', split="val")
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

def save_img(img, img_path, is_image):
    if is_image:
        img[:,0,:,:] += 72.39
        img[:,1,:,:] += 82.91
        img[:,2,:,:] += 73.16
        img /= 255
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
        # out = out.data.max(1)[1][0]
        out = torch.nn.functional.softmax(out, 1)[:,1,:,:]
        save_img(out, 'images/out'+str(i)+'.png', False)

        i += 1
        if i>10:
            break
