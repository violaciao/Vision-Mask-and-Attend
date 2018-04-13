from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from config import *

## to keep a track of your network on tensorboard, set USE_TENSORBOARD TO 1 in config file.

if USE_TENSORBOARD:
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=TENSORBOARD_SERVER)
    try:
        cc.remove_experiment(EXP_NAME)
    except:
        pass
    foo = cc.create_experiment(EXP_NAME)


## to use the GPU, set GPU_MODE TO 1 in config file

use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.set_device(CUDA_DEVICE)

count=0

### SECTION 1 - data loading and shuffling/augmentation/normalization : all handled by torch automatically.

# use imagenet dataset's mean and standard deviation to normalize their dataset approximately. These numbers are imagenet mean and standard deviation!

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomSizedCrop(224),
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

## the structure looks like this : 
# data_dir
#      |- train 
#            |- roses
#                 |- rose_image_1
#                 |- rose_image_2
#                        .....

#            |- sunflowers
#                 |- sunflower_image_1
#                 |- sunflower_image_2
#                        .....
#            |- lilies
#                 |- lilies_image_1
#                        .....
#      |- val
#            |- roses
#            |- sunflowers
#            |- lilies

data_dir = DATA_DIR
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val'] 
         if (not x.startswith('.DS_Store')) and (not x.startswith('dummy'))}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=25)
                for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes


### SECTION 2 : Writing the functions that do training and validation phase. 

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=100):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                mode='train'
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()
                mode='val'

            running_loss = 0.0
            running_corrects = 0

            counter=0
            # Iterate over data.
            for data in dset_loaders[phase]:
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())
                    except:
                        print(inputs,labels)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)
                # print('loss done')                
                # Just so that you can keep track that something's happening and don't feel like the program isn't running.
                if counter%100==0:
                    print("Reached iteration ", counter)
                counter+=1

                # backward + optimize only if in training phase
                if phase == 'train':
                    # print('loss backward')
                    loss.backward()
                    # print('done loss backward')
                    optimizer.step()
                    # print('done optim')
                # print evaluation statistics
                try:
                    running_loss += loss.data[0]
                    running_corrects += torch.sum(preds == labels.data)
                except:
                    print('unexpected error, could not calculate loss or do a sum.')
            print('trying epoch loss')
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


            # deep copy the model
            if phase == 'val':
                if USE_TENSORBOARD:
                    foo.add_scalar_value('epoch_loss',epoch_loss,step=epoch)
                    foo.add_scalar_value('epoch_acc',epoch_acc,step=epoch)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)
                    print('new best accuracy = ',best_acc)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('returning and looping back')
    return best_model

# This function changes the learning rate over the training model.
def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (DECAY_WEIGHT**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


### SECTION 3 : DEFINING MODEL ARCHITECTURE.

# Set the pre-trained model in the ocnfig file by MODEL_FT.
if MODEL_FT == 'inception_v3':
    model_ft = models.inception_v3(pretrained=True)   
elif MODEL_FT == 'resnet18':
    model_ft = models.resnet18(pretrained=True)
elif MODEL_FT == 'resnet152':
    model_ft = models.resnet152(pretrained=True)
elif MODEL_FT == 'vgg16':
    model_ft = models.vgg16(pretrained=True)
elif MODEL_FT == 'densenet':
    model_ft = models.densenet161(pretrained=True)
elif MODEL_FT == 'alexnet':
    model_ft = models.alexnet(pretrained=True)
else:
    model_ft = None
    print("Error from FT Config Setting! Invalid Pretrained Model Name Value!")


# Set model parameters
if 'vgg' in MODEL_FT or 'alexnet' in MODEL_FT:
    num_ftrs = model_ft.classifier[6].in_features
    features = list(model_ft.classifier.children())[:-1]
    features.apppend([nn.Linear(num_ftrs, NUM_CLASSES)])
    model_ft.classifier = nn.Sequential(*features)

elif 'densenet' in MODEL_FT:
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, NUM_CLASSES)

else:
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)


criterion = nn.CrossEntropyLoss()

if use_gpu:
    criterion.cuda()
    model_ft.cuda()

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)



# Run the functions and save the best model in the function model_ft.
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=NUM_EPOCHS)

# Save model in CPU
torch.save(model_ft.cpu().state_dict(), MODEL_CPU_SAVING_PATH)





