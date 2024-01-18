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
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Create DataLoader for validation
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

#Load the pretrained vgg-16 model
resnet_model=models.resnet18(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in resnet_model.parameters():
    param.requires_grad = False

# Modify the last fully connected layer to match the number of classes in your dataset
num_classes = len(train_dataset.classes)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.01)

# Specify the number of training epochs
num_epochs = 5  # Change this based on your needs

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = resnet_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 2 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')