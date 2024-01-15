'''
Author -Aditya Bhatt
Objective - Alexnet POC
Notes -
1.
'''

#Import the library
import torch
import torch.nn as nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

#Define Data Transformations
transform=transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
])

#Define the training dataset
train_dataset=datasets.ImageFolder(root='./Lenet-5-POC/Data/train',transform=transform)

#Define the validation dataset
val_dataset=datasets.ImageFolder(root='./Lenet-5-POC/Data/val',transform=transform)

# Create DataLoader for training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create DataLoader for validation
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def Alexnet(num_classes):
    model=nn.Sequential(
        #First Convolution Layer
        nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3,stride=2),

        # Second Convolutional Layer
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        
        # Third Convolutional Layer
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        
        # Fourth Convolutional Layer
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        
        # Fifth Convolutional Layer
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        
        # Adaptive Average Pooling Layer
        nn.AdaptiveAvgPool2d((6, 6)),
        
        # Flatten Layer
        nn.Flatten(),
        
        # First Fully Connected Layer
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        
        # Second Fully Connected Layer
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        
        # Output Layer
        nn.Linear(4096, num_classes)  # Assuming binary classification, adjust num_classes accordingly
    )

    return model

#Training Loop
def train(model,train_loader,criterion,optimizer,num_epochs=1):
    for epoch in range(num_epochs):
        for batch_idx,(data, target) in enumerate(train_loader):
            # Clear the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Compute the loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
            # Print training progress
            if batch_idx % 100 == 0:
                with open('training_log_Alexnet.txt', 'a') as log_file:
                 log_file.write(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}\n')


        
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(Alexnet(num_classes=2).parameters(), lr=0.01)

# Specify the number of training epochs
num_epochs = 100

# Training loop
train(Alexnet(2), train_loader, criterion, optimizer, num_epochs)

