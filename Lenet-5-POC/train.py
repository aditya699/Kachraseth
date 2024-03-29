'''
Author -Aditya Bhatt 12:00PM 31/12/2023
Objective - 
1.Create a mock poc for lenet 5 as a part of a creating a real time waste management system
Notes-
1.Resize+Converts the image data to a PyTorch tensor and normalizes the pixel values to the range [0, 1].
2.Data loader will go through the entire training dataset in batches of 32 images,
and the order of images within each batch will be random.
'''
#Import Libraries
import torch
from  torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets
import torch.nn as nn

#Tranform the dataset
transform=transforms.Compose([
transforms.Resize((32,32)),
    transforms.ToTensor(),
])

#Define the training Dataset
train_dataset =datasets.ImageFolder(root='Lenet-5-POC/Data/train',transform=transform)

# Define validation dataset
val_dataset = datasets.ImageFolder(root='Lenet-5-POC/Data/val', transform=transform)

# Create DataLoader for training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create DataLoader for validation
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#Define the lenet-5 model
def lenet(num_classes):
    model=nn.Sequential(
        nn.Conv2d(3,6,kernel_size=5,stride=1,padding=0),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16*5*5, 120),
        nn.Linear(120, 84),
        nn.Linear(84, num_classes)
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
                with open('training_log_lenet.txt', 'a') as log_file:
                 log_file.write(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}\n')


        
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lenet(num_classes=2).parameters(), lr=0.001)

# Specify the number of training epochs
num_epochs = 100

# Training loop
train(lenet(2), train_loader, criterion, optimizer, num_epochs)


