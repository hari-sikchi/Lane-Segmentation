import sys
sys.path.append("/mnt/data/tejus/Lane-Segmentation/datasets/tusimple")
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from config import *
from  augmentations import *
#from datasets.kitti.kitti_loader import kittiLoader
#from datasets.tusimple.tusimple_loader import tusimpleLoader
from CAN_lfe import *
# import visdom
import math
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tusimple_numpy_loader import tusimpleLoader
from progress.bar import ChargingBar
import os
import cv2
from PIL import Image

def save_img(img, img_path, is_image):
	if is_image:
		img[:,0,:,:] += 72.39 #why?
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

#can put decay
def update_lr(optimizer, epoch):
	if epoch == 25:
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.1

	if epoch == 40:
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.1

####################### HYPERPARAMETER #############################
BATCH_SIZE = 1
EXP_NAME = 'OVERFIT2'#'exp-23-5-19'
RUN_STRING = 'runs/OVERFIT2'#'runs/exp-23-5-19'
LEARNING_RATE = 0.00032
NUM_EPOCH = 100
####################################################

# Weights path check
try:
	if EXP_NAME not in os.listdir('weights/'):
		os.mkdir('weights/' + EXP_NAME)
	
except:
	print("Make directory issue")

print("Made Save Directory")


# writer for tensorboardX
writer = SummaryWriter(RUN_STRING)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(102)

net = CAN()

# net.load_state_dict(torch.load('./weights/'+EXP_NAME+'/trained.wts'), strict=False)
net.to(device)

params = net.state_dict()

# initialization of dilation layers kernels

# dilated_conv_layers = [35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57]
# for layer_idx in dilated_conv_layers:
# 	# print (layer_idx)
# 	w = params['features.'+str(layer_idx)+'.weight']
# 	b = params['features.'+str(layer_idx)+'.bias']
# 	w.fill_(0)
# 	for i in range(w.shape[0]):
# 		w[i,i,1,1] = 1
# 	#print(w)

# 	b.fill_(0)

# 	params['features.'+str(layer_idx)+'.weight'] = w 
# 	params['features.'+str(layer_idx)+'.weight'] = b

# #torch.save(net.state_dict(), 'test_identity.wts')
# layer_idx = 61 #56
# w = params['features.'+str(layer_idx)+'.weight']
# w.fill_(0)
# for i in range(w.shape[0]):
# 	w[i,i,0,0] = 1

# params['features.'+str(layer_idx)+'.weight'] = w

# adding augmentation to dataset
augmentations = Compose([RandomRotate(5), RandomHorizontallyFlip()])

print("Loading Data")
'''
train_dataset = tusimpleLoader('/mnt/data/tejus/test_set/', split="test",channel_split=True, augmentations=augmentations)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
'''
train_dataset = tusimpleLoader('/mnt/data/tejus/train_set/', split="train",channel_split=True, augmentations=augmentations)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)

val_dataset = tusimpleLoader('/mnt/data/tejus/train_set/', split="val",channel_split=True, augmentations=augmentations)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)

# dataset = kittiLoader('/mnt/data/tejus/kitti_road/data_road/', split="train", augmentations=augmentations)
# trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, pin_memory=True)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = LEARNING_RATE)
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = 0.0001, momentum=0.9) # 0.00001
pos_weight = torch.Tensor([57]).cuda() #should actually be 57
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# loss_fn = nn.CrossEntropyLoss()

best_val_loss = math.inf

print("Starting training")
k = 37#2703

bar = ChargingBar('Processing', max=NUM_EPOCH, suffix='%(percent).2f%% - %(elapsed).2f%% - %(eta).2f%%')
for EPOCHS in range(NUM_EPOCH):
	
	optimizer.zero_grad()
	net.train()
	update_lr(optimizer, EPOCHS)
	running_loss = 0
	for i, data in enumerate(trainloader):
		imgs, labels = data
		# img = imgs[0].cpu().numpy()
		# img = img
		
		# img = img.astype(np.uint8)
		# print("PIKA")
		# print(img[0,3,4])
		# img = img.transpose([1,2,0])
		# img = img.reshape(720,1280,3)
		# cv2.imshow('a', img)
		# cv2.waitKey(0)
		# print(imgs.shape)
		imgs = imgs.float().cuda()
		imgs = imgs*1.0/255.0
		labels = labels.float().cuda()
		# imgs, labels = imgs.type(torch.FloatTensor).to(device), labels.type(torch.FloatTensor).to(device)
		# print(imgs.shape)
		# img = imgs[0].cpu().numpy()
		# print("CHU")
		# print(img[0,3,4])
		# img = img*255
		# img = img.astype(np.uint8)
		# img = img.reshape(720,1280,3)
		# img = img.reshape(720, 1280, 3)
		# print(np.max(img))
		# print(np.min(img))
		

		# cv2.imshow('a', img)
		# cv2.waitKey(0)
		# print(imgs)
		
		# print("PIKA")
		# print(imgs)
		out = net(imgs)
		# print("CHU")
		# print(out)
		
		loss = loss_fn(out, labels)
		loss.backward()
		running_loss += loss.item()

		if((i+1)%4==0):
			optimizer.step()
			optimizer.zero_grad()
			writer.add_scalar('loss/training_loss', running_loss, k + 1)
			k+=1
			print("EPOCH %d, iteration %d: loss=%f"%(EPOCHS, (2*i + 1), running_loss))
			running_loss = 0
			torch.save(net.state_dict(), 'weights/' + EXP_NAME + '/trained.wts')
	
	net.eval()
	running_loss = 0
	count = 0
	with torch.no_grad():
		for i, data in enumerate(valloader):
			imgs, labels = data
			imgs, labels = imgs.type(torch.FloatTensor).to(device), labels.type(torch.FloatTensor).to(device)
			imgs = imgs/255.0
			# print('Mean of images')
			# temp = imgs[1]
			# temp = np.transpose(temp, [1,2,0])
			# cv2.imshow('Display', temp)
			# cv2.waitKey(0)
			# plt.imshow(temp)
			# plt.imshow()
			# break
			# print(imgs[0])
			# print(np.mean(imgs[0]), axis = None)
			# print(imgs)
			out = net(imgs)
			temp = out.cpu()
			temp = np.array(temp)
			# print(temp[:,0,:,:])
			# print(temp[:,1,:,:])
			# print(np.mean(temp[:,1,:,:]))
			z = np.argmax(temp, axis = 1)
			z = z*255
			# print(np.max(z))
			for temp1 in z:
				count += 1
				# break
				img = Image.fromarray(temp1, mode="1")
				img.save("/mnt/data/tejus/Lane-Segmentation/CAN_LFE/eval_images_output/" +EXP_NAME +'/'+ str(count) + ".png")
			# break
			loss = loss_fn(out, labels)
			running_loss += loss.item()
			writer.add_scalar('loss/validation_loss', loss, k + 1)

	print("EPOCH %d: VAL loss=%f"%(EPOCHS, running_loss))
	bar.next()
	
	if running_loss < best_val_loss:
		best_val_loss = running_loss
		torch.save(net.state_dict(), 'best_val_recent.wts')


	

bar.finish()