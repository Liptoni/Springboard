# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 18:08:58 2018

@author: Ian
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy

if __name__ == "__main__":           
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
    
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        print('Learning Rate:', lr, 'Batch Size:', batch_size)
        for epoch in range(num_epochs):
            time_elapsed = time.time() - since
            print('Epoch {}/{}, {:.0f}m:{:.0f}s'.format(epoch+1, num_epochs, time_elapsed // 60, time_elapsed % 60))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Learning Rate:', lr, 'Batch Size:', batch_size,'Best val Acc: {:4f}'.format(best_acc))
        print('')
        results.append([lr, batch_size, best_acc]) #used this for testing different batch sized and initial learning rates
        
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model    

    torch.manual_seed(24)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(24)

    
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    #get the images dataset
    data_dir = 'Images'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                      data_transforms[x])
                              for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    #use GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    batch_sizes = [20]#[4, 8, 10, 12]; [15, 20, 25
    learn_rates = [0.001]#[0.1, 0.01, 0.001]
    for batch_size in batch_sizes:
                  
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                     shuffle=True, num_workers=4)#
                      for x in ['train', 'val']}        
    
        ##Pick the pre-trained model to use
               
        #model_ft = models.resnet18(pretrained=True)
        #model_ft = models.resnet34(pretrained=True)
        model_ft = models.resnet101(pretrained=True)
        #model_ft = models.resnet152(pretrained=True)
    
        #reset the output layer to handle the number of categories I have        
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 4)
        
        model_ft = model_ft.to(device)
        
        #instantiate loss criterion
        criterion = nn.CrossEntropyLoss()
        for lr in learn_rates:  
            learning_rate = lr
            # Observe that all parameters are being optimized
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
            
            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
            
            model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                   num_epochs=25)
            
            #save model if desired            
            save_model=False
            if save_model is True:
                #saves only parameters
                torch.save(model_ft.state_dict(), 'Saved_Models/PhotoClass_DenseNet161_TL.pkl')        
    
    #print results of testing
    for run in results:
        print('Learning Rate: {} Batch Size: {} Best Acc {:4f}'.format(run[0], run[1], run[2]))
        print('-'*10)
