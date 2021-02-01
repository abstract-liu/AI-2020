import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import sampler

from model.resnet import *

import argparse
import numpy as np
from matplotlib import pyplot as plt


dtype = torch.float32
USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_iteration_every = 5
print_valid_every = 1

print('using device:', device)

BATCHSIZE = 64
NEPOCH = 10
LR = 0.0001
STEP = 7


#preprocess img
#transform
transform_train=transforms.Compose([
	transforms.Resize((256,256)),
	transforms.RandomCrop((224,224)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

transform_val=transforms.Compose([ 
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])
#load 
train_set = dset.ImageFolder('./dataset/dataset-resized', transform = transform_train)
valid_set = dset.ImageFolder('./dataset/dataset-resized', transform = transform_val)

#dataloader 
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCHSIZE, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCHSIZE)

#load and modify model
model = resnet50tuned()
state_dict = torch.load('model/best_model_v2.pth')
model.load_state_dict(state_dict)
model.to(device=device)

#optimizer
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=0.1)

#learning progress statistic data
train_curve = list()

def train():
    for epoch in range(NEPOCH):
        print('epoch %d '%(epoch))
        model.train()
        
        iter_samples = 0
        iter_correct = 0


        for batch_idx, (img,label) in enumerate(train_loader):
            img = img.to(device=device, dtype=dtype)
            label = label.to(device=device, dtype=torch.long)
        
            #compute output and loss
            scores = model(img)
            loss = F.cross_entropy(scores, label)
        
            #compute gradient and back
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #add loss to history
            train_curve.append(loss.item())

            #check accuracy
            if batch_idx % print_iteration_every == 0:
                _, predicted = scores.max(1)
                iter_correct += (predicted == label).sum()
                iter_samples += label.size(0)
                acc = float(iter_correct)/iter_samples 
                print('Iteration %d, loss = %.4f, acc = %.2f' % (batch_idx, loss.item(), 100 * acc))
        scheduler.step() 
        


def test():       
    val_samples = 0
    val_correct = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(valid_loader):
            img = img.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            label = label.to(device=device, dtype=torch.long)
            scores = model(img)

            _, preds = scores.max(1)
            val_correct += (preds == label).sum()
            val_samples += preds.size(0)

    acc = float(val_correct) / val_samples
    print('Valid Got %d / %d, acc = (%.2f)' % (val_correct, val_samples, 100 * acc))

            
#train()
#torch.save(model.state_dict(), './model/best_model_v6.pth')
test()

#plot visualization
plt.plot(range(len(train_curve)), train_curve, label='train loss')
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.show()


