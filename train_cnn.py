"""
Custom Image Dataset CNN Training Script
H200 GPU Cluster Workshop

Trains a CNN on any custom image dataset organized in folders.
Works with the data structure shown in the workshop:
    data/
        train/
            class1/
            class2/
        test/
            class1/
            class2/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
import argparse


# CNN Model Definition
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 4 pooling layers: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Convolutional layers with pooling
        x = self.pool(self.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(self.relu(self.conv2(x)))  # 112 -> 56
        x = self.pool(self.relu(self.conv3(x)))  # 56 -> 28
        x = self.pool(self.relu(self.conv4(x)))  # 28 -> 14
        
        # Flatten
        x = x.view(-1, 256 * 14 * 14)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
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


def evaluate(model, loader, criterion, device):
    """Evaluate the model"""
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


def main():
    parser = argparse.ArgumentParser(description='Train CNN on custom image dataset')
    parser.add_argument('--data', type=str, default='data', help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("-" * 60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("Loading dataset...")
    train_dataset = datasets.ImageFolder(
        root=os.path.join(args.data, 'train'),
        transform=train_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(args.data, 'test'),
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Get class names and number of classes
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("-" * 60)
    
    # Initialize model
    model = CustomCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Print progress
        print(f"Epoch [{epoch}/{args.epochs}]  "
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
                'class_names': class_names,
                'num_classes': num_classes,
                'train_acc': train_acc,
                'test_acc': test_acc,
            }, 'models/custom_cnn.pth')
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Model saved to: models/custom_cnn.pth")
    print("-" * 60)
    
    # Save training metrics
    with open('results/cnn_training_metrics.txt', 'w') as f:
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Classes: {', '.join(class_names)}\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Training time: {elapsed_time:.1f} seconds\n")
        f.write(f"Best test accuracy: {best_acc:.2f}%\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")


if __name__ == '__main__':
    main()
