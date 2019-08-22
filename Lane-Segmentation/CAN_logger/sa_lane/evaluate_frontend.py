import sys
sys.path.append('/home/tejus/lane-seg-experiments/Segmentation/')

import torch
import torchvision
from torchvision import datasets, models, transforms
from datasets.tusimple.config import CONFIG
from datasets.kitti.augmentations import *
from datasets.kitti.kitti_loader import kittiLoader
from datasets.tusimple.tusimple_loader import tusimpleLoader
from CAN_frontend import CAN
import matplotlib.pyplot as plt
from metrics import runningScore 

device = torch.device("cpu")

# dataset = kittiLoader('/mnt/data/tejus/kitti_road/data_road/', split="train")
dataset = tusimpleLoader('/home/tejus/Downloads/train_set/', split="train")
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
checkpoint_dir = '/home/tejus/lane-seg-experiments/Segmentation/CAN_logger/frontend_only/runs/2018-10-06_14-51-26/best_val_model.pkl'
checkpoint = torch.load(checkpoint_dir)


import matplotlib.pyplot as plt
# def save_img(img, img_path, is_image):
#     if is_image:
#         img[:,0,:,:] *= 0.229
#         img[:,1,:,:] *= 0.224
#         img[:,2,:,:] *= 0.225

#         img[:,0,:,:] += 0.485
#         img[:,1,:,:] += 0.456
#         img[:,2,:,:] += 0.406

#         img = img[:,[1,0,2],:,:]
#         print(torch.min(img), torch.max(img))
#     img = torchvision.utils.make_grid(img)
#     npimg = img.cpu().numpy()
#     #img = Image.fromarray(np.transpose(npimg,(1,0,2)),'BGR')
#     print(np.transpose(npimg,(1,2,0)).shape)
#     #npimg=np.array(npimg,dtype=np.uint8)
#     img = Image.fromarray(np.transpose(npimg,(1,2,0)),'RGB')
#     img.save(img_path)
#     # npimg[npimg > 0.1] = 1
#     # npimg[npimg <= 0.1] = 0
#     #plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     #plt.savefig(img_path)

def save_img(img, img_path, is_image):
    if is_image:
        img[:,0,:,:] *= 0.229
        img[:,1,:,:] *= 0.224
        img[:,2,:,:] *= 0.225

        img[:,0,:,:] += 0.485
        img[:,1,:,:] += 0.456
        img[:,2,:,:] += 0.406

        img = img[:,[1,0,2],:,:]
        #print(torch.min(img), torch.max(img))
    img = torchvision.utils.make_grid(img)
    npimg = img.cpu().numpy()

    plt.imsave(img_path, np.transpose(npimg, (1, 2, 0)), format="png", cmap="hot")

    # fig = plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.axis('off')
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    # plt.savefig(img_path, bbox_inches='tight', pad_inches = 0)


    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.savefig(img_path)

net = CAN()
#net.load_state_dict(torch.load('best_val_tensorboard.wts'))
net.load_state_dict(checkpoint["model_state"])
net.to(device)
net.eval()

i = 0

save_data = True


running_metrics_val =runningScore(2)


with torch.no_grad():
    for data in trainloader:
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        if save_data:
            save_img(imgs, 'images/'+str(i)+'.png', True)
        
        out = net(imgs)
        
        # For BCE Loss
        # out = torch.nn.functional.sigmoid(out)
        # out = torch.where(out > 0.2, torch.Tensor([1]).cuda(), torch.Tensor([0]).cuda())
        # out = out.data.max(1)[1]
        # plt.imshow(out[0][1])
        # plt.savefig('images/out' + str(i) + '.png')
        pred = out.data.max(1)[1]
        print(pred)
        out = torch.nn.functional.softmax(out, 1)[:,1,:,:]
        #out = out.data.max(0)[1]
        out = torch.where(out > 0.2, torch.Tensor([1]), torch.Tensor([0]))
        running_metrics_val.update(pred.cpu().numpy(),labels.cpu().numpy())
        if save_data:
            save_img(out, 'images/out'+str(i)+'.png', False)

        #visualize using pyplot
        # plt.imshow(out[0])
        #plt.savefig('images/out' + str(i) + '.png')
          
        

        i += 1
        print(i)
        if i>10:
            break

    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)
    running_metrics_val.reset()
