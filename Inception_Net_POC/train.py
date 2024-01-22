import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.models as models

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 input size
    transforms.ToTensor(),
])

# Define the training dataset
train_dataset = datasets.ImageFolder(root='Inception_Net_POC/Data/train', transform=transform)

# Create DataLoader for training
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the validation dataset
val_dataset = datasets.ImageFolder(root='Inception_Net_POC/Data/val', transform=transform)

# Create DataLoader for validation
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
optimizer = torch.optim.Adam(inception_model.parameters(), lr=0.01)

# Training Loop with Training Accuracy Calculation
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Extract only logits from the InceptionOutputs tuple
            output, _ = model(data)
            logits = output  # No need to extract logits, assuming output is logits
            
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = logits.max(1)
            total_correct += predicted.eq(target).sum().item()
            total_samples += target.size(0)

            if batch_idx % 2 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Calculate and print training accuracy after each epoch
        accuracy = total_correct / total_samples * 100
        print(f'Training Accuracy after Epoch {epoch + 1}: {accuracy:.2f}%')

        # Save the trained model after all epochs
    torch.save(model.state_dict(), 'inception_model_state_dict.pth')

    # Call the validate function after training
    validate(model, val_loader, criterion)

# Validation Function
def validate(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)

            # Calculate validation accuracy
            _, predicted = output.max(1)
            total_correct += predicted.eq(target).sum().item()
            total_samples += target.size(0)

        accuracy = total_correct / total_samples * 100
        print(f'Validation Accuracy: {accuracy:.2f}%')

# Specify the number of training epochs
num_epochs = 5

# Training loop with validation
train(inception_model, train_loader, val_loader, criterion, optimizer, num_epochs)
