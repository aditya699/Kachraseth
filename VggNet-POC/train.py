'''
Author        -    Aditya Bhatt
Objective     -    Implementing VGG-16 Using Transfer Learning
General Notes -     
1.The requires_grad attribute in PyTorch indicates whether the gradients for a parameter should be computed during backpropagation. 
By setting it to False, you are essentially telling PyTorch not to update the weights of these layers during training. 
This is often referred to as "freezing" the layers.In transfer learning scenarios, freezing the layers is common when you want to use a 
pre-trained model but fine-tune it for a specific task with a smaller dataset. Since the early layers of the model have learned generic features, you may not want to update them too much and risk losing those valuable features.
'''
#Import Libraries
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

#Define Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

#Load the pretrained vgg-16 model
vgg16_model=models.vgg16(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in vgg16_model.parameters():
    param.requires_grad = False

# Modify the last fully connected layer to match the number of classes in your dataset
num_classes = 2  # Change this based on your dataset
vgg16_model.classifier[-1] = nn.Linear(4096, num_classes)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16_model.parameters(), lr=0.001)

# Specify the number of training epochs
num_epochs = 5  # Change this based on your needs

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = vgg16_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 2 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')