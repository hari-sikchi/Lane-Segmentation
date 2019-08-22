import os
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader 
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter

import sys
sys.path.append('/home/tejus/lane-seg-experiments/Segmentation/')
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from datasets.kitti.config import CONFIG
from datasets.kitti.kitti_loader import kittiLoader
from datasets.tusimple.tusimple_loader import tusimpleLoader
from CAN_sa  import CAN
from datasets.tusimple.augmentations import *
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from metrics import runningScore 
from datetime import datetime
import math

# Definitions

TRAIN_BATCH = 1
VAL_BATCH = 4
resume_training = False  
checkpoint_dir = '/home/tejus/lane-seg-experiments/Segmentation/CAN_logger/frontend_only/runs/2018-10-06_14-51-26/best_val_model_tested.pkl'
run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logdir = os.path.join('runs/' , str(run_id))
writer = SummaryWriter(log_dir=logdir)
print('RUNDIR: {}'.format(logdir))
logger = get_logger(logdir)
logger.info('Let the party begin')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(102)

# Network definition 

net = CAN()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = 0.0001, momentum=0.9) # 0.00001
# loss_fn = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 2, verbose = True, min_lr = 0.000001)
loss_fn = nn.CrossEntropyLoss()
net.to(device)

if not resume_training:
    pretrained_state_dict = models.vgg16(pretrained=True).state_dict()
    # remove classifier layers from pretrained weights
    pretrained_state_dict = {k:v for k, v in pretrained_state_dict.items() if 'classifier' not in k}
    #rename parameters. ordering changes because maxpool layers aren't present in network.
    pretrained_state_dict['features.22.weight'] = pretrained_state_dict.pop('features.21.weight')
    pretrained_state_dict['features.22.bias'] = pretrained_state_dict.pop('features.21.bias')
    net.load_state_dict(pretrained_state_dict, strict=False)
    start_iter = 0
else:
    checkpoint = torch.load(checkpoint_dir)
    net.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_iter = checkpoint["epoch"]
    logger.info(
        "Loaded checkpoint '{}' (epoch {})".format(
            checkpoint_dir, checkpoint["epoch"]
        )
    )

net.train()

augmentations = Compose([RandomRotate(5), RandomHorizontallyFlip()])
train_dataset = tusimpleLoader('/home/tejus/Downloads/train_set/', split="train", augmentations=augmentations)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True, num_workers=TRAIN_BATCH, pin_memory=True)

val_dataset = tusimpleLoader('/home/tejus/Downloads/train_set/', split="val", augmentations=None)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=VAL_BATCH, shuffle=True, num_workers=VAL_BATCH, pin_memory=True)

running_metrics_val = runningScore(2)
best_val_loss = math.inf
val_loss = 0
ctr = 0
best_iou=-100
val_loss_meter = averageMeter()
time_meter = averageMeter()

for EPOCHS in range(start_iter, 50):

    # Training

    net.train()
    running_loss = 0
    for i, data in enumerate(trainloader):
        start_ts = time.time()
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        
        out = net(imgs)
        
        loss = loss_fn(out, labels)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        time_meter.update(time.time() - start_ts)

        if i%100 == 99:
            print("EPOCH %d: %d/3000 loss=%f Time/Image=%.4f "%(EPOCHS, TRAIN_BATCH*(i+1), loss.item(),time_meter.avg/TRAIN_BATCH))
            #print("EPOCH %d: %d/3000 loss=%f Time/Image=%.4f "%(EPOCHS, TRAIN_BATCH*(i+1), running_loss/(100),time_meter.avg/TRAIN_BATCH))
            time_meter.reset()

    fmt_str = "Epoch {:d}:  Loss: {:.4f} "
    print_str = fmt_str.format(EPOCHS + 1, 
                                ((running_loss/len(trainloader.dataset))*TRAIN_BATCH))
    print(print_str)
    logger.info(print_str)
                                
    writer.add_scalar('loss/training_loss',((running_loss/len(trainloader.dataset))*TRAIN_BATCH), EPOCHS+1)


    for param_group in optimizer.param_groups:
        print('Current learning rate is: ', param_group['lr'])

    # Validation

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(valloader):
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            
            out = net(imgs)
            loss = loss_fn(out, labels)
            ctr+=1
            pred = out.data.max(1)[1]
            running_metrics_val.update(pred.cpu().numpy(),labels.cpu().numpy())
            val_loss_meter.update(loss.item())
            val_loss += loss.item()

    writer.add_scalar('loss/val_loss', val_loss_meter.avg, EPOCHS+1)
    logger.info("Epoch %d Loss: %.4f" % (EPOCHS+1, val_loss_meter.avg))
    print("[Validation]Epoch %d Loss: %.4f" % (EPOCHS+1, val_loss_meter.avg))
    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)
        logger.info('{}: {}'.format(k, v))
        writer.add_scalar('val_metrics/{}'.format(k), v, EPOCHS+1)

    for k, v in class_iou.items():
        logger.info('{}: {}'.format(k, v))
        writer.add_scalar('val_metrics/cls_{}'.format(k), v, EPOCHS+1)

    running_metrics_val.reset()
    val_loss_meter.reset()


    #print("EPOCH %d: VAL loss=%f"%(EPOCHS, val_loss/(ctr)))
    
    # scheduler.step(val_loss)

    # if score["Mean IoU : \t"] >= best_iou:
    #     best_iou = score["Mean IoU : \t"]
    #     torch.save(net.state_dict(), 'best_val.wts')

    if score["Mean IoU : \t"] >= best_iou:
        best_iou = score["Mean IoU : \t"]
        state = {
            "epoch": EPOCHS + 1,
            "model_state": net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_iou": best_iou,
        }
        save_path = os.path.join(writer.file_writer.get_logdir(),
                                    "best_val_model.pkl")
        torch.save(state, save_path)

    # if val_loss < best_val_loss:
    #     torch.save(net.state_dict(), 'best_val.wts')

    #torch.save(net.state_dict(), 'trained.wts')
