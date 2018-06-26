# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 21:50:00 2018

@author: Ian
"""
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from datetime import datetime
import copy

#need this to run on Windows b/c of multiple workers in data loaders
if __name__ == "__main__":
    StartTime = datetime.now()
    
    torch.manual_seed(24)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(24)
        
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
    
    #load testing set
    test_dataset = datasets.ImageFolder(root='Images/testing/', transform=data_transforms['val'])
    
    '''
    train_dataset values are tupples. 
    First element is a Tensor with size (3,224,224)
    Second element is the label
    '''
    
    #set batch size and number of iterations (we want 5 epochs)
    batch_size = 20
    n_iters = 320
    
    #calc the number of epochs
    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)
    
    
    #create an iterable for training dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)#
    
    #create an iterable for validation dataset
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)#
    
    #create an iterable for testing dataset
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10, shuffle=False, num_workers=4)
    
    
    #Create model class
    class CNN_Model(nn.Module):
        def __init__(self):
            super(CNN_Model, self).__init__()
            
            #convolution 1
            self.cnn1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
# =============================================================================
#             self.bn1 = nn.BatchNorm2d(64)
# =============================================================================
            self.relu1 = nn.ReLU()
            
            #Pooling 1
            self.maxpool1 = nn.MaxPool2d(kernel_size=2) #to do average pooling use nn.AvgPool2d(kernel_size=2)
# =============================================================================
#             self.avgpool1 = nn.AvgPool2d(kernel_size=2)
# =============================================================================
            
            #convolution 2
            self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5,  stride=1, padding=2)
# =============================================================================
#             self.bn2 = nn.BatchNorm2d(128)
# =============================================================================
            self.relu2 = nn.ReLU()
            
            #Pooling 2
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)
# =============================================================================
#             self.avgpool2 = nn.AvgPool2d(kernel_size=2)
# =============================================================================
            
            #fully Connected 1 (readout)
            self.fc1 = nn.Linear(128 * 56 * 56, 4)
            
        def forward(self, x):
            #conv 1
            out = self.cnn1(x)
# =============================================================================
#             out = self.bn1(out)
# =============================================================================
            out = self.relu1(out)
            
            #Pooling 1
            out = self.maxpool1(out)
# =============================================================================
#             out = self.avgpool1(out)
# =============================================================================
            
            #conv 2
            out = self.cnn2(out)
# =============================================================================
#             out = self.bn2(out)
# =============================================================================
            out = self.relu2(out)
            
            #Pooling 2
            out = self.maxpool2(out)
# =============================================================================
#             out = self.avgpool2(out)
# =============================================================================
            
            #Resize
            '''
            current size: (20, 128, 56, 56) --20 immages per batch, each image is now 128 x 56 x 56 after conv and pooling
            out.size(0): 20
            New out size: (20, 128*56*56)
            '''
            out = out.view(out.size(0), -1)
            
            #readout
            out = self.fc1(out)
            return out
            
    model = CNN_Model()
    
    if torch.cuda.is_available():
        model.cuda()
        
    #instantiate loss class
    criterion = nn.CrossEntropyLoss()
    
    #instantiate optimizer class
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            
    #train the model
    iter = 0
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        scheduler.step()
        for i, (images, labels) in enumerate(train_loader):
            
            #load images    
            if torch.cuda.is_available():      
                images = images.cuda()
                labels = labels.cuda()
            
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
            
            iter += 1
            
            if iter % 20 == 0:
                #Calc accuracy
                correct = 0
                total = 0
                #iterate through test dataset
                for images, labels in val_loader:
                    if torch.cuda.is_available():      
                        images = images.cuda()
                        
                    outputs = model(images)
                    #get predictions from max value
                    _, predicted = torch.max(outputs.data, 1)
                    
                    #total number of labels
                    total += labels.size(0)
                    
                    if torch.cuda.is_available():
                        #have to bring back to cpu to use sum. Convert to np array or else sums are off
                        correct += (predicted.cpu() == labels.cpu()).numpy().sum()            
                    else:
                        correct += (predicted == labels).numpy().sum()
    
                accuracy =100. *  correct / total
                #make sure to save best possible model by checking if current accuracy is better than previous iterations
                if accuracy >=   best_acc:
                    best_acc = accuracy
                    best_model_weights = copy.deepcopy(model.state_dict())
                #print loss
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data.item(), accuracy))
    
    print('Best val Acc: {:4f}'.format(best_acc))
    #load the best model weights for saving
    model.load_state_dict(best_model_wts)

    #save model if desired
    if save_model is True:
        #saves only parameters
        torch.save(model.state_dict(), 'Saved_Models/PhotoClass_Simple_CNN_Model.pkl')
    
           
    print('Elapsed:', datetime.now()-StartTime)