"""
MNIST Neural Network Training Script
H200 GPU Cluster Workshop Demo

Trains a simple feedforward neural network on MNIST digits.
Outputs match the format shown in the workshop slides.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10
NUM_WORKERS = 2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
print("-" * 60)

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Download and load datasets
print("Loading MNIST dataset...")
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print("-" * 60)


# Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


# Initialize model, loss, and optimizer
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("-" * 60)


# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# Training loop
print("Starting training...")
start_time = time.time()
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    # Print progress (matching workshop slide format)
    print(f"Epoch [{epoch}/{EPOCHS}]  "
          f"Loss: {train_loss:.4f}  "
          f"Acc: {train_acc:.1f}%  "
          f"Test Loss: {test_loss:.4f}  "
          f"Test Acc: {test_acc:.1f}%")
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, 'models/mnist_nn.pth')

# Training complete
elapsed_time = time.time() - start_time
print("-" * 60)
print(f"Training complete in {elapsed_time:.1f} seconds")
print(f"Best test accuracy: {best_acc:.2f}%")
print(f"Model saved to: models/mnist_nn.pth")
print("-" * 60)

# Final evaluation
print("\nFinal Test Set Evaluation:")
final_loss, final_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {final_loss:.4f}")
print(f"Test Accuracy: {final_acc:.2f}%")

# Save training metrics
with open('results/training_metrics.txt', 'w') as f:
    f.write(f"Training completed in {elapsed_time:.1f} seconds\n")
    f.write(f"Best test accuracy: {best_acc:.2f}%\n")
    f.write(f"Final test accuracy: {final_acc:.2f}%\n")
    f.write(f"Total epochs: {EPOCHS}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")

print("\nTraining metrics saved to: results/training_metrics.txt")
