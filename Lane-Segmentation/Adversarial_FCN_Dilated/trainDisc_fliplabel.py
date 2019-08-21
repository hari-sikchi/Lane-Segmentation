import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import os, scipy.io
from scipy import misc
from skimage import io
from datasets.config import CONFIG
from datasets.kitti_loader import kittiLoader
from torch.autograd import Variable
from torch.autograd import grad

from FCN_Dilated import FCN_Dilated
from netdSemSeg import Discriminator

from loss import cross_entropy2d
from datasets.augmentations import *
from torch.utils import data
import visdom

import scipy.misc as m
import matplotlib.pyplot as plt

import torchvision.utils as vutils
import random

gpu = 0

device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def lr_decayer(optimizerG, optimizerD, num_epoch):
    cur_epoch = num_epoch
    lr_g = 0.0002 #0.00005
    lr_d = 0.0002
    minlr = 0.00005
    decay_interval = 20
    decay_level = 0.5
    while cur_epoch >= decay_interval:
        lr_g = lr_g * decay_level
        lr_d = lr_d * decay_level
        cur_epoch -= decay_interval
        lr_g = max(lr_g, minlr)
        lr_d = max(lr_d, minlr)
    for param_group in optimizerG.param_groups:
        param_group['lr'] = lr_g
        print ("New learning rate: ", lr_g)
    for param_group in optimizerD.param_groups:
        param_group['lr'] = lr_d
        print ("New learning rate: ", lr_d)

batch_size = 1
netG = FCN_Dilated()
netD = Discriminator(gpu)

netG.load_state_dict(torch.load('vgg_fcn_pretrained.wts'))
netG.to(device)
netD.to(device)

netG.train()
netD.train()

netD.apply(weights_init)
# Initialize test image
# mean = mean = np.array([72.39, 82.91, 73.16])
# test_image = m.imread('um_000001.png')
# test_image = np.array(test_image, dtype=np.uint8)
# test_image = test_image[:, :, [1, 0, 2]]

# test_image = test_image.astype(np.float64)
# test_image -= mean
# test_image = test_image.transpose(2, 0, 1)
# test_image = torch.from_numpy(test_image).float()
# test_image = test_image.unsqueeze(0).cuda()

ctr = 0

# Freeze first 4 layers
for w in netG.parameters():
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
dataset = kittiLoader('data_road/', split="train", augmentations=augmentations)
trainloader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

optimizerG = torch.optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr=0.0002, betas=(0.5,0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5,0.999))



def train_dcgan():
    loss_fn = nn.CrossEntropyLoss()

    virtual_batch_size = 4
    best_loss = 1000.0
    best_loss_cr = 1000.0
    best_loss_epoch = 0
    best_loss_cr_ep = 0
    errD = 1000.0
    errG_adv = Variable(torch.Tensor([1000.0]))
    errG_cr = Variable(torch.Tensor([1000.0]))
    errG = Variable(torch.Tensor([1000.0]))

    real_label = 1
    fake_label = 0

    criterion = nn.BCEWithLogitsLoss()
    updateD = True
    adv_loss_wt = 0.1    
    flippos = False
    for EPOCHS in range(100):
        running_loss = 0
        running_loss_cr = 0
        running_adv = 0
        flip = 0
        

        if EPOCHS > 1 and EPOCHS % 2 == 0:
                updateD = not updateD
        if updateD:
            print ("Updating D")
        else:
            print ("Updating G")

        for i, data in enumerate(trainloader):
            flip += 1

            if flip > 2:
                real_label = 0
                fake_label = 1
                flip = 0

            imgs, labels = data
            imgs, labels = imgs.to(device).float(), labels.to(device)
            # lab0 = labels.clone()
            # lab0[labels==1] = 2
            # lab0[labels==0] = 1
            # lab0[lab0==2] = 0
            # labelsD = torch.cat((lab0, labels), dim=0)
            # labelsD = Variable(labelsD)
            
            label = torch.full((batch_size,), real_label, device=device)

            # Update Discriminator netD
            
            # print (labels.size())
            # labelsD = labels.unsqueeze(0).float()
            if updateD:
                netD.zero_grad()
                # outputD_real = netD(labelsD.unsqueeze(0).float())
                outputD_real = netD(labels.unsqueeze(0).float())

                errD_real = criterion(outputD_real, label)
                errD_real.backward()
                D_x = outputD_real.mean().item()

            # Train D on FCN generated output
            outSeg = netG(imgs).float()
            index = outSeg.max(1)[1]
            fakeInputD = (outSeg[0][1] * index.float()).unsqueeze(0).float()

            if updateD:
                label.fill_(fake_label)
                output = netD(fakeInputD.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()
            
            # Update G (FCN_Dilated)
            if not updateD:
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                outputD_fake = netD(fakeInputD)
                errG_adv = criterion(outputD_fake, label)
                errG_cr = loss_fn(outSeg, labels)
                errG = (adv_loss_wt * errG_adv) + errG_cr
                errG.backward()
                D_G_z2 = outputD_fake.mean().item()
                optimizerG.step()

            real_label = 1
            fake_label = 0

            if show_visdom:
                vis.line(X=torch.ones((1,1)).cpu()*i + EPOCHS * len(trainloader),
                                        Y= torch.Tensor([errG.data[0]]).unsqueeze(0).cpu(),win = loss_window,update='append' )
            # if i%virtual_batch_size == 0 or i == len(dataset)-1:

            running_loss += errG.item()
            running_loss_cr += errG_cr.item()
            running_adv += errG_adv.item()
            # if i%10 == 0:
            #     print(str(i)+'/230')

        torch.save(netG.state_dict(), 'trained.wts')
        torch.save(netD.state_dict(), 'trainD.wts')

        pred = outSeg.data.max(1)[1].cpu().numpy()[0]
    # plt.imshow(pred)
    # plt.savefig(str(EPOCHS) + '_out.png')
        plt.imshow(pred)
        plt.savefig('output/' + str(EPOCHS) + '_out.png')

        vutils.save_image(imgs.squeeze(0).cpu().data, 'output/' + str(EPOCHS) + '_orig.png', normalize=True)

        
        vutils.save_image(imgs.squeeze(0).cpu().data, str(EPOCHS) + '_orig.png', normalize=True)

        if (running_loss / 230.0) < best_loss:
            best_loss = running_loss/230.0
            best_loss_epoch = EPOCHS
            torch.save(netG.state_dict(), 'best_loss_dcgan.wts')
        if (running_loss_cr / 230.0) < best_loss_cr:
            best_loss_cr = running_loss_cr / 230.0
            best_loss_cr_ep = EPOCHS
            torch.save(netG.state_dict(), 'best_seg_dcgan.wts')

        print("Epoch %d: LossG = %f ErrGadv = %f ErrG_cr = %f LossD = %f Best_Loss = %f Best_Loss_Epoch = %d" % (EPOCHS, running_loss/230.0, running_adv/230.0, running_loss_cr/230.0, errD, best_loss, best_loss_epoch))    # lr_decayer(optimizer, EPOCHS)
        print('Best Seg iter', best_loss_cr_ep, 'Best Seg Loss', best_loss_cr)


    torch.save(netG.state_dict(), 'trained_dcgan.wts')
    torch.save(netD.state_dict(), 'trainD_dcgan.wts')

def train_wgan():
    loss_fn = nn.CrossEntropyLoss()

    virtual_batch_size = 4
    best_loss = 1000.0
    best_loss_cr = 1000.0
    best_loss_epoch = 0
    best_loss_cr_ep = 0
    errD = 0
    errG_adv = Variable(torch.Tensor([0]))
    errG_cr = Variable(torch.Tensor([0]))
    errG = Variable(torch.Tensor([0]))

    real_label = 1
    fake_label = 0

    criterion = nn.BCELoss()
    updateD = True
    adv_loss_wt = 1.0

    for EPOCHS in range(100):
        running_loss = 0
        running_loss_cr = 0
        running_adv = 0

        # if EPOCHS > 1 and EPOCHS % 3 == 0:
        #         updateD = not updateD
        # if updateD:
        #     print ("Updating D")
        # else:
        #     print ("Updating G")

        for i, data in enumerate(trainloader):
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            lab0 = labels.clone()
            # lab0[labels==1] = 2
            # lab0[labels==0] = 1
            # lab0[lab0==2] = 0
            # labelsD = torch.cat((lab0, labels), dim=0)
            # labelsD = Variable(labelsD)
            
            label = torch.full((batch_size,), real_label, device=device)

            # Update Discriminator netD
            
            # print (labels.size())
            # labelsD = labels.unsqueeze(0).float()
            netD.zero_grad()
            # outputD_real = netD(labelsD.unsqueeze(0).float())
            outputD_real = netD(labels.unsqueeze(0).float())


            D_x = outputD_real.mean().item()

            # Train D on FCN generated output
            outSeg = netG(imgs)
            index = outSeg.max(1)[1]
            fakeInputD = (outSeg[0][1] * index.float()).unsqueeze(0)

            label.fill_(fake_label)
            output = netD(fakeInputD.float().detach())
            D_G_z1 = output.mean().item()
            errD = -torch.mean(output - outputD_real) # errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            for p in netD.parameters():
                p.data.clamp_(-0.05, 0.05)

            # Update G (FCN_Dilated)
            if ((i+1) % 5) == 0:
                netG.zero_grad()
                outSeg = netG(imgs)
                index = outSeg.max(1)[1]
                fakeInputD = (outSeg[0][1] * index.float()).unsqueeze(0)

                label.fill_(real_label)  # fake labels are real for generator cost
                outputD_fake = netD(fakeInputD.float())
                outputD_real = netD(labels.unsqueeze(0).float())
                errG_adv = -torch.mean(outputD_fake)
                errG_cr = loss_fn(outSeg.float(), labels)
                errG = (adv_loss_wt * errG_adv) + errG_cr
                errG.backward()
                D_G_z2 = outputD_fake.mean().item()
                optimizerG.step()

            if show_visdom:
                vis.line(X=torch.ones((1,1)).cpu()*i + EPOCHS * len(trainloader),
                                        Y= torch.Tensor([errG.data[0]]).unsqueeze(0).cpu(),win = loss_window,update='append' )
            # if i%virtual_batch_size == 0 or i == len(dataset)-1:

            running_loss += errG.item()
            running_loss_cr += errG_cr.item()
            running_adv += errG_adv.item()
            # if i%10 == 0:
            #     print(str(i)+'/230')
            
        torch.save(netG.state_dict(), 'trained.wts')
        torch.save(netD.state_dict(), 'trainD.wts')

        pred = outSeg.data.max(1)[1].cpu().numpy()[0]
    # plt.imshow(pred)
    # plt.savefig(str(EPOCHS) + '_out.png')
        plt.imshow(pred)
        plt.savefig('output/' + str(EPOCHS) + '_out.png')

        vutils.save_image(imgs.squeeze(0).cpu().data, 'output/' + str(EPOCHS) + '_orig.png', normalize=True)

        
        vutils.save_image(imgs.squeeze(0).cpu().data, str(EPOCHS) + '_orig.png', normalize=True)

        if (running_loss / 230.0) < best_loss:
            best_loss = running_loss/230.0
            best_loss_epoch = EPOCHS
            torch.save(netG.state_dict(), 'best_loss_wgan.wts')
        if (running_loss_cr / 230.0) < best_loss_cr:
            best_loss_cr = running_loss_cr / 230.0
            best_loss_cr_ep = EPOCHS
            torch.save(netG.state_dict(), 'best_seg_wgan.wts')

        print("Epoch %d: LossG = %f ErrGadv = %f ErrG_cr = %f LossD = %f Best_Loss = %f Best_Loss_Epoch = %d" % (EPOCHS, running_loss/230.0, running_adv/230.0, running_loss_cr/230.0, errD, best_loss, best_loss_epoch))    # lr_decayer(optimizer, EPOCHS)
        print('Best Seg iter', best_loss_cr_ep, 'Best Seg Loss', best_loss_cr)
        lr_decayer(optimizerG, optimizerD, EPOCHS)


    torch.save(netG.state_dict(), 'trained_wgan.wts')
    torch.save(netD.state_dict(), 'trainD_wgan.wts')

def train_wgan_gp():
    loss_fn = nn.CrossEntropyLoss()

    virtual_batch_size = 4
    best_loss = 1000.0
    best_loss_cr = 1000.0
    best_loss_epoch = 0
    best_loss_cr_ep = 0
    errD = 0
    errG_adv = Variable(torch.Tensor([0]))
    errG_cr = Variable(torch.Tensor([0]))
    errG = Variable(torch.Tensor([0]))

    real_label = 1
    fake_label = 0

    criterion = nn.BCELoss()
    updateD = True
    adv_loss_wt = 1.0

    for EPOCHS in range(100):
        running_loss = 0
        running_loss_cr = 0
        running_adv = 0

        # if EPOCHS > 1 and EPOCHS % 3 == 0:
        #         updateD = not updateD
        # if updateD:
        #     print ("Updating D")
        # else:
        #     print ("Updating G")

        for i, data in enumerate(trainloader):
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            lab0 = labels.clone()
            # lab0[labels==1] = 2
            # lab0[labels==0] = 1
            # lab0[lab0==2] = 0
            # labelsD = torch.cat((lab0, labels), dim=0)
            # labelsD = Variable(labelsD)
            
            label = torch.full((batch_size,), real_label, device=device)

            # Update Discriminator netD
            
            # print (labels.size())
            # labelsD = labels.unsqueeze(0).float()
            netD.zero_grad()
            # outputD_real = netD(labelsD.unsqueeze(0).float())
            outputD_real = netD(labels.unsqueeze(0).float())


            D_x = outputD_real.mean().item()

            # Train D on FCN generated output
            outSeg = netG(imgs)
            index = outSeg.max(1)[1]
            fakeInputD = (outSeg[0][1] * index.float()).unsqueeze(0)

            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update


            label.fill_(fake_label)
            output = netD(fakeInputD.float())
            D_G_z1 = output.mean().item()
            alpha = torch.rand((batch_size, 1, 1, 1))
            alpha = alpha.to(device)

            x_hat = alpha * labels.float().data + (1 - alpha) * fakeInputD.float().data
            x_hat.requires_grad = True

            pred_hat = netD(x_hat)
            gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            
            gradient_penalty = 1.0 * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

            errD = torch.mean(output - outputD_real) + gradient_penalty # errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            for p in netD.parameters():
                p.requires_grad = False # to avoid computation

            # Update G (FCN_Dilated)
            if ((i+1) % 5) == 0:
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                outSeg = netG(imgs)
                index = outSeg.max(1)[1]
                fakeInputD = (outSeg[0][1] * index.float()).unsqueeze(0)
                outputD_fake = netD(fakeInputD.float())
                outputD_real = netD(labels.unsqueeze(0).float())
                errG_adv = -torch.mean(outputD_fake)
                errG_cr = loss_fn(outSeg.float(), labels)
                errG = (adv_loss_wt * errG_adv) + errG_cr
                errG.backward()
                D_G_z2 = outputD_fake.mean().item()
                optimizerG.step()

            if show_visdom:
                vis.line(X=torch.ones((1,1)).cpu()*i + EPOCHS * len(trainloader),
                                        Y= torch.Tensor([errG.data[0]]).unsqueeze(0).cpu(),win = loss_window,update='append' )
            # if i%virtual_batch_size == 0 or i == len(dataset)-1:

            running_loss += errG.item()
            running_loss_cr += errG_cr.item()
            running_adv += errG_adv.item()
            # if i%10 == 0:
            #     print(str(i)+'/230')
            
        torch.save(netG.state_dict(), 'trained.wts')
        torch.save(netD.state_dict(), 'trainD.wts')

        pred = outSeg.data.max(1)[1].cpu().numpy()[0]
    # plt.imshow(pred)
    # plt.savefig(str(EPOCHS) + '_out.png')
        plt.imshow(pred)
        plt.savefig('output/' + str(EPOCHS) + '_out.png')

        vutils.save_image(imgs.squeeze(0).cpu().data, 'output/' + str(EPOCHS) + '_orig.png', normalize=True)

        if (running_loss / 230.0) < best_loss:
            best_loss = running_loss/230.0
            best_loss_epoch = EPOCHS
            torch.save(netG.state_dict(), 'best_loss_wgangp.wts')
        if (running_loss_cr / 230.0) < best_loss_cr:
            best_loss_cr = running_loss_cr / 230.0
            best_loss_cr_ep = EPOCHS
            torch.save(netG.state_dict(), 'best_seg_wgangp.wts')

        print("Epoch %d: LossG = %f ErrGadv = %f ErrG_cr = %f LossD = %f Best_Loss = %f Best_Loss_Epoch = %d" % (EPOCHS, running_loss/230.0, running_adv/230.0, running_loss_cr/230.0, errD, best_loss, best_loss_epoch))    # lr_decayer(optimizer, EPOCHS)
        print('Best Seg iter', best_loss_cr_ep, 'Best Seg Loss', best_loss_cr)
        lr_decayer(optimizerG, optimizerD, EPOCHS)


    torch.save(netG.state_dict(), 'trained_wgangp.wts')
    torch.save(netD.state_dict(), 'trainD_wgangp.wts')

if __name__=="__main__":
    train_dcgan()


