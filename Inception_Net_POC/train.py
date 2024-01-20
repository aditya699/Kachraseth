import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 input size
    transforms.ToTensor(),
])

# Define the training dataset
train_dataset = datasets.ImageFolder(root='./Lenet-5-POC/Data/train', transform=transform)

# Create DataLoader for training
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Load the pre-trained InceptionV3 model
inception_model = models.inception_v3(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in inception_model.parameters():
    param.requires_grad = False

# Modify the last fully connected layer to match the number of classes in your dataset
num_classes = 2  # Change this based on your dataset
inception_model.fc = nn.Linear(inception_model.fc.in_features, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(inception_model.parameters(), lr=0.001)

# Training Loop with Training Accuracy Calculation
def train(model, train_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output, _ = model(data)  # Extract logits from the InceptionOutputs object
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = output.max(1)
            total_correct += predicted.eq(target).sum().item()
            total_samples += target.size(0)

            if batch_idx % 2 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Calculate and print training accuracy after each epoch
        accuracy = total_correct / total_samples * 100
        print(f'Training Accuracy after Epoch {epoch + 1}: {accuracy:.2f}%')

        # Save the trained model after all epochs
    torch.save(inception_model.state_dict(), 'inception_model_state_dict.pth')

# Specify the number of training epochs
num_epochs = 5

# Training loop
train(inception_model, train_loader, criterion, optimizer, num_epochs)
