"""
Transfer Learning with ResNet-18
H200 GPU Cluster Workshop

Uses a pre-trained ResNet-18 model and fine-tunes it on custom datasets.
Much faster training and better accuracy than training from scratch.
Pre-trained on ImageNet (1.2M images, 1000 classes).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time
import argparse


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
    parser = argparse.ArgumentParser(description='Transfer learning with ResNet-18')
    parser.add_argument('--data', type=str, default='data', help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (10 is usually enough)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=224, help='Image size (224 for ResNet)')
    parser.add_argument('--freeze-layers', type=int, default=7, 
                        help='Number of layer groups to freeze (0-8, default 7)')
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
    
    # Data transforms (using ImageNet statistics)
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    
    # Get class information
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("-" * 60)
    
    # Load pre-trained ResNet-18
    print("Loading pre-trained ResNet-18...")
    print("Downloading weights from ImageNet (1.2M images, 1000 classes)...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze early layers (feature extraction)
    if args.freeze_layers > 0:
        layers_to_freeze = [
            model.conv1,
            model.bn1,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            model.fc
        ]
        
        for i, layer in enumerate(layers_to_freeze[:args.freeze_layers]):
            for param in layer.parameters():
                param.requires_grad = False
        
        print(f"Frozen first {args.freeze_layers} layer groups")
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print("-" * 60)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.lr)
    
    # Training loop
    print("Starting transfer learning...")
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
                'architecture': 'resnet18',
            }, 'models/resnet18_transfer.pth')
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Model saved to: models/resnet18_transfer.pth")
    print("-" * 60)
    
    # Save training metrics
    with open('results/transfer_training_metrics.txt', 'w') as f:
        f.write(f"Architecture: ResNet-18 (pre-trained on ImageNet)\n")
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Classes: {', '.join(class_names)}\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Frozen layers: {args.freeze_layers}/8\n")
        f.write(f"Training time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)\n")
        f.write(f"Best test accuracy: {best_acc:.2f}%\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
    
    print("\nTip: For better accuracy, try:")
    print("  - More epochs (15-20)")
    print("  - Lower learning rate (0.0001)")
    print("  - Unfreeze more layers (--freeze-layers 5 or 3)")


if __name__ == '__main__':
    main()
