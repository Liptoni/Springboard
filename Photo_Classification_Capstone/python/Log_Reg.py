# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 21:08:38 2018

@author: Ian
"""

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from datetime import datetime

if __name__ == "__main__":
    StartTime = datetime.now()

    #set up transformations for training and validation datasets
    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
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
    
    #load training dataset
    train_dataset = datasets.ImageFolder(root='Images/train/', transform=data_transforms['train'])
    
    #load validation dataset 
    val_dataset = datasets.ImageFolder(root='Images/val/', transform=data_transforms['val'])
    
    
    
    #set batch size and number of iterations
    batch_size = 20
    n_iters = 160
    
    #calc number of epochs
    num_epochs = n_iters / (len(train_dataset)/batch_size)
    num_epochs = int(num_epochs)
    
    
    #create an iterable for training dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)#
    
    #create an iterable for validation dataset
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)#
    
    
    #create a Log Reg class
    class LogisticRegressionModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(LogisticRegressionModel, self).__init__()
                self.linear = nn.Linear(input_dim, output_dim)
                
            def forward(self, x):
                out = self.linear(x)
                return out
            
            
    #instantiate model
    input_dim = 3*224*224
    output_dim = 4
    
    model = LogisticRegressionModel(input_dim, output_dim)
    
    #move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()
    
    #instantiate loss class
    criterion = nn.CrossEntropyLoss()
    
    #instantiate optimizer
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    n_iter=0
    for epoch in range(num_epochs):
        scheduler.step()
        for i, (images, labels) in enumerate(train_loader):
            #load images
            if torch.cuda.is_available():
                images = images.view(-1, 3*224*224).cuda()            
                labels = labels.cuda()    
            else:
                images = images.view(-1, 3*224*224)
                labels = labels
                
            #clear gradients wrt parameters
            optimizer.zero_grad()
            
            
            #Forward pass to get output/logits
            outputs = model(images)
            

            #calculate loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            
            
            #get gradients wrt parameters
            loss.backward()
            
            #update parameters
            optimizer.step()
            
            n_iter += 1
            #check on model accuracy
            if n_iter % 20 == 0:
                correct = 0
                total = 0
                
                #iterate through validation dataset
                for images, labels in val_loader:
                    if torch.cuda.is_available():
                        images = images.view(-1, 3*224*224).cuda()     
                    else:
                        images = images.view(-1, 3*224*224)
                        
                    outputs = model(images)
                    
                    #get predicitions from model
                    _, predicted = torch.max(outputs.data, 1)
                    
                    #total number of images
                    total += labels.size(0)
                    
                    
                    if torch.cuda.is_available():
                        #need to move back to CPU to do sums
                        correct += (predicted.cpu() == labels.cpu()).numpy().sum() 
                    else:
                        correct += (predicted == labels).numpy().sum() 
                
                #calc accuracy
                accuracy = 100 * correct / total
                
                #print loss
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(n_iter, loss.data.item(), accuracy))
            
    save_model=False
    if save_model is True:
        #saves only parameters
        torch.save(model.state_dict(), 'Saved_Models/PhotoClass_Logreg_Model.pkl')
    
           
    print('Elapsed:', datetime.now()-StartTime)     
                  
            
            
