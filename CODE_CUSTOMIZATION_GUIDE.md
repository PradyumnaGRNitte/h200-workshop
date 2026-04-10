# 🔧 Code Customization Guide

**How to modify the Python training scripts for your specific needs**

---

## 📋 Table of Contents

1. [When to Customize vs When to Use Flags](#when-to-customize-vs-when-to-use-flags)
2. [Understanding the Script Structure](#understanding-the-script-structure)
3. [Common Customizations by Script](#common-customizations-by-script)
4. [Model Architecture Modifications](#model-architecture-modifications)
5. [Data Processing Customizations](#data-processing-customizations)
6. [Training Loop Modifications](#training-loop-modifications)
7. [Advanced Customizations](#advanced-customizations)
8. [Creating Your Own Script](#creating-your-own-script)

---

## 1. When to Customize vs When to Use Flags

### ✅ Use Command-Line Flags (NO code editing needed):

```bash
# These DON'T require editing code:
python3 train_transfer.py \
    --data data \              # Data folder
    --epochs 20 \              # Number of epochs
    --batch-size 32 \          # Batch size
    --lr 0.001 \               # Learning rate
    --img-size 224 \           # Image size
    --freeze-layers 7          # How many layers to freeze
```

**Use flags for:**
- Changing hyperparameters (epochs, batch size, learning rate)
- Adjusting image size
- Pointing to different data folders
- Controlling layer freezing (transfer learning)

---

### ✏️ Edit Code When You Need To:

**Must edit code for:**
- ✅ Custom model architecture (different layers, filters)
- ✅ Different loss functions (not CrossEntropy)
- ✅ Custom data augmentation
- ✅ Different optimizers (not Adam)
- ✅ Multi-task learning (multiple outputs)
- ✅ Custom preprocessing (beyond standard normalization)
- ✅ Adding callbacks/checkpointing strategies
- ✅ Custom metrics (beyond accuracy)

---

## 2. Understanding the Script Structure

### All Three Scripts Follow This Pattern:

```
┌─────────────────────────────────────────┐
│ 1. Imports                              │
│    - torch, torch.nn, torchvision       │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 2. Model Definition (Class)             │
│    - __init__: Define layers            │
│    - forward: Define data flow          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 3. Training Functions                   │
│    - train_epoch: One epoch training    │
│    - evaluate: Test set evaluation      │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 4. Main Function                        │
│    - Parse arguments                    │
│    - Setup data loaders                 │
│    - Initialize model                   │
│    - Training loop                      │
│    - Save results                       │
└─────────────────────────────────────────┘
```

---

## 3. Common Customizations by Script

### 📄 train_nn_mnist.py (Simple Neural Network)

**File Location:** `~/my_project/train_nn_mnist.py`

#### Customization 1: Change Hidden Layer Sizes

**Original (Lines 16-20):**
```python
class SimpleNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)        # Hidden layer
        self.fc3 = nn.Linear(64, num_classes) # Output layer
```

**Customized (Larger network):**
```python
class SimpleNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # ← Changed from 128
        self.fc2 = nn.Linear(256, 128)       # ← Changed from 64
        self.fc3 = nn.Linear(128, num_classes)
```

**When to do this:** When you need more model capacity for complex patterns

---

#### Customization 2: Add More Layers

**Original (Lines 16-26):**
```python
class SimpleNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
```

**Customized (4 layers instead of 3):**
```python
class SimpleNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)      # ← NEW layer
        self.fc4 = nn.Linear(64, num_classes)  # ← Renamed from fc3
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))         # ← NEW layer in forward
        x = self.dropout(x)
        x = self.fc4(x)                    # ← Renamed from fc3
        return x
```

---

### 📄 train_cnn.py (Custom CNN)

**File Location:** `~/my_project/train_cnn.py`

#### Customization 1: Change Number of Filters

**Original (Lines 27-36):**
```python
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
```

**Customized (More filters for complex images):**
```python
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers - MORE FILTERS
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)   # ← 64 instead of 32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # ← 128 instead of 64
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)# ← 256 instead of 128
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)# ← 512 instead of 256
```

**IMPORTANT:** If you change filters in conv layers, you MUST update the fully connected layer:

**Original (Line 43):**
```python
self.fc1 = nn.Linear(256 * 14 * 14, 512)  # 256 from conv4
```

**Must change to:**
```python
self.fc1 = nn.Linear(512 * 14 * 14, 512)  # ← 512 now from conv4
```

**When to do this:** For high-resolution images or complex patterns that need more feature maps

---

#### Customization 2: Change Input Image Size

**Problem:** The script expects 224x224 images. What if yours are 128x128 or 512x512?

**Original (Lines 42-45):**
```python
# Fully connected layers
# After 4 pooling layers: 224 -> 112 -> 56 -> 28 -> 14
self.fc1 = nn.Linear(256 * 14 * 14, 512)
```

**For 128x128 images:**
```python
# After 4 pooling layers: 128 -> 64 -> 32 -> 16 -> 8
self.fc1 = nn.Linear(256 * 8 * 8, 512)  # ← Changed from 14 * 14
```

**For 512x512 images:**
```python
# After 4 pooling layers: 512 -> 256 -> 128 -> 64 -> 32
self.fc1 = nn.Linear(256 * 32 * 32, 512)  # ← Changed from 14 * 14
```

**Formula:** 
```
final_size = input_size / (2^num_pooling_layers)
For 4 pooling layers: final_size = input_size / 16

Examples:
- 224 / 16 = 14
- 128 / 16 = 8
- 512 / 16 = 32
```

---

#### Customization 3: Grayscale Instead of RGB

**Original (Line 32):**
```python
self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 = RGB channels
```

**Customized (for grayscale images like X-rays):**
```python
self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # ← 1 channel for grayscale
```

**Also update normalization (Line 147):**

**Original:**
```python
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # RGB means/stds
```

**Customized:**
```python
transforms.Normalize([0.5], [0.5])  # ← Single channel for grayscale
```

---

### 📄 train_transfer.py (Transfer Learning)

**File Location:** `~/my_project/train_transfer.py`

#### Customization 1: Use Different Pre-trained Model

**Original (Line 152):**
```python
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
```

**Options:**

**Smaller/Faster Models:**
```python
# MobileNetV2 - Very fast, less accurate
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
# Update final layer:
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# EfficientNet-B0 - Good balance
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# Update final layer:
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
```

**Larger/More Accurate Models:**
```python
# ResNet-50 - Better accuracy, slower
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# Update final layer (same as ResNet-18):
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# ResNet-101 - Best accuracy, much slower
model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
```

**For Medical Images (pretrained on medical data):**
```python
# You'd need to download specific medical imaging models
# Example: RadImageNet pretrained models
# See: https://github.com/BMEII-AI/RadImageNet
```

---

#### Customization 2: Fine-tune More or Fewer Layers

**Original (Lines 155-169):**
```python
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
```

**Strategy Guide:**

```python
# STRATEGY 1: Freeze almost everything (very small dataset < 500 images)
--freeze-layers 7  # Only train final FC layer
# Fastest training, least overfitting

# STRATEGY 2: Balanced (500-2000 images)
--freeze-layers 5  # Train layer4 + layer3 + FC
# Good balance

# STRATEGY 3: Fine-tune everything (> 5000 images)
--freeze-layers 0  # Train all layers
# Slowest, but best accuracy if you have data
```

**To freeze specific layers only (advanced):**
```python
# Freeze only conv1 and layer1
for param in model.conv1.parameters():
    param.requires_grad = False
for param in model.layer1.parameters():
    param.requires_grad = False

# Everything else will train
```

---

## 4. Model Architecture Modifications

### Adding Batch Normalization

**Why:** Improves training stability and speed

**In train_cnn.py, add after each conv layer:**

**Original:**
```python
self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
```

**With BatchNorm:**
```python
self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
self.bn1 = nn.BatchNorm2d(32)  # ← ADD THIS

self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
self.bn2 = nn.BatchNorm2d(64)  # ← ADD THIS
```

**Update forward function:**
```python
def forward(self, x):
    x = self.pool(self.relu(self.bn1(self.conv1(x))))  # ← Add bn1
    x = self.pool(self.relu(self.bn2(self.conv2(x))))  # ← Add bn2
    # ... rest of forward pass
```

---

### Adding Residual Connections

**Why:** Helps with deep networks (prevents vanishing gradients)

**Add to train_cnn.py:**

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # ← Residual connection
        out = self.relu(out)
        return out

# Use in your model:
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.block1 = ResidualBlock(3, 32)
        self.block2 = ResidualBlock(32, 64)
        self.pool = nn.MaxPool2d(2, 2)
        # ... rest of model
```

---

## 5. Data Processing Customizations

### Custom Data Augmentation

**Location:** Line 141-148 in train_cnn.py or train_transfer.py

**Original:**
```python
train_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

**Heavy Augmentation (for small datasets):**
```python
train_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # ← ADD
    transforms.RandomRotation(30),     # ← Increase from 10
    transforms.RandomAffine(          # ← ADD
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.ColorJitter(
        brightness=0.3,                # ← Increase
        contrast=0.3,                  # ← Increase
        saturation=0.3,                # ← ADD
        hue=0.1                        # ← ADD
    ),
    transforms.RandomGrayscale(p=0.1), # ← ADD
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

**Medical Images (X-rays, CT scans):**
```python
train_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.RandomHorizontalFlip(),  # OK for some medical images
    # NO vertical flip - not realistic for X-rays
    # NO heavy color jitter - intensity matters
    transforms.RandomRotation(5),      # Small rotation only
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Grayscale normalization
])
```

---

### Custom Dataset Class

**When:** Your data isn't in standard ImageFolder format

**Add before main() function:**

```python
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class CustomDataset(Dataset):
    """
    For CSV-based datasets like:
    image_path,label
    img1.jpg,0
    img2.jpg,1
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(self.data.iloc[idx, 1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Use in main():
train_dataset = CustomDataset(
    csv_file='data/train.csv',
    root_dir='data/images',
    transform=train_transform
)
```

---

## 6. Training Loop Modifications

### Adding Learning Rate Scheduler

**Why:** Reduce learning rate when loss plateaus

**Add after optimizer definition (around line 197):**

```python
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ADD THIS:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      # Reduce when loss stops decreasing
    factor=0.5,      # Reduce by half
    patience=3,      # Wait 3 epochs
    verbose=True
)
```

**Update training loop (around line 212):**

```python
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    # ADD THIS:
    scheduler.step(test_loss)  # Update learning rate based on test loss
    
    print(f"Epoch [{epoch}/{args.epochs}]  ...")
```

---

### Early Stopping

**Why:** Stop training when test accuracy stops improving

**Add before training loop:**

```python
# Early stopping setup
best_acc = 0.0
patience = 5  # Stop if no improvement for 5 epochs
patience_counter = 0

for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = train_epoch(...)
    test_loss, test_acc = evaluate(...)
    
    # Early stopping logic
    if test_acc > best_acc:
        best_acc = test_acc
        patience_counter = 0
        # Save model
        torch.save(...)
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        print(f"Best accuracy: {best_acc:.2f}%")
        break
```

---

### Saving Checkpoints Every N Epochs

**Add in training loop:**

```python
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = train_epoch(...)
    test_loss, test_acc = evaluate(...)
    
    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        checkpoint_path = f'models/checkpoint_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'test_acc': test_acc,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
```

---

## 7. Advanced Customizations

### Multi-Task Learning (Multiple Outputs)

**Example:** Predict both disease type AND severity

**Modify model:**

```python
class MultiTaskCNN(nn.Module):
    def __init__(self, num_disease_classes, num_severity_classes):
        super(MultiTaskCNN, self).__init__()
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # ... more conv layers
        
        # Task-specific heads
        self.disease_fc = nn.Linear(512, num_disease_classes)
        self.severity_fc = nn.Linear(512, num_severity_classes)
    
    def forward(self, x):
        # Shared features
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Two outputs
        disease_pred = self.disease_fc(x)
        severity_pred = self.severity_fc(x)
        
        return disease_pred, severity_pred

# Update training loop:
def train_epoch_multitask(model, loader, criterion, optimizer, device):
    model.train()
    for images, (disease_labels, severity_labels) in loader:
        images = images.to(device)
        disease_labels = disease_labels.to(device)
        severity_labels = severity_labels.to(device)
        
        disease_pred, severity_pred = model(images)
        
        loss_disease = criterion(disease_pred, disease_labels)
        loss_severity = criterion(severity_pred, severity_labels)
        loss = loss_disease + loss_severity  # Combined loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

### Using Different Loss Functions

**Original (Line 196):**
```python
criterion = nn.CrossEntropyLoss()
```

**For Imbalanced Classes:**
```python
# Calculate class weights
class_counts = [800, 200, 50]  # Example: 800 normal, 200 disease1, 50 disease2
total = sum(class_counts)
class_weights = torch.tensor([total/count for count in class_counts])
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**For Binary Classification (2 classes):**
```python
# Option 1: CrossEntropy (works fine)
criterion = nn.CrossEntropyLoss()

# Option 2: Binary Cross Entropy (requires model output change)
criterion = nn.BCEWithLogitsLoss()
# Must change final layer to output 1 value instead of 2
```

**For Regression (predicting continuous values):**
```python
criterion = nn.MSELoss()  # Mean Squared Error
# Or
criterion = nn.L1Loss()   # Mean Absolute Error
```

---

### Custom Optimizer Settings

**Original (Line 189-190):**
```python
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=args.lr)
```

**Different Optimizers:**

**SGD with Momentum:**
```python
optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=0.9,
    weight_decay=1e-4  # L2 regularization
)
```

**AdamW (better than Adam for many cases):**
```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=0.01
)
```

**Different Learning Rates for Different Layers:**
```python
# Fine-tune with different LRs
optimizer = optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-5},   # Very small LR
    {'params': model.conv2.parameters(), 'lr': 1e-4},   # Small LR
    {'params': model.fc.parameters(), 'lr': 1e-3}       # Normal LR
])
```

---

## 8. Creating Your Own Script

### Template for New Custom Script

**Save as `train_my_custom_model.py`:**

```python
"""
My Custom Training Script
Description: [What this does]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
import time


# ============== MODEL DEFINITION ==============
class MyCustomModel(nn.Module):
    """
    Your custom architecture
    """
    def __init__(self, num_classes):
        super(MyCustomModel, self).__init__()
        
        # Define your layers here
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Add more layers...
        
        self.fc = nn.Linear(64 * 56 * 56, num_classes)  # Adjust size
    
    def forward(self, x):
        # Define forward pass
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # Add more operations...
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


# ============== TRAINING FUNCTION ==============
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


# ============== EVALUATION FUNCTION ==============
def evaluate(model, loader, criterion, device):
    """Evaluate on test set"""
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


# ============== MAIN FUNCTION ==============
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train custom model')
    parser.add_argument('--data', type=str, default='data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(args.data, 'train'),
        transform=train_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(args.data, 'test'),
        transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=2)
    
    # Get number of classes
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    model = MyCustomModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("Starting training...")
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, 
                                            criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch [{epoch}/{args.epochs}]  "
              f"Loss: {train_loss:.4f}  "
              f"Acc: {train_acc:.1f}%  "
              f"Test Acc: {test_acc:.1f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
            }, 'models/my_custom_model.pth')
    
    print(f"\nBest test accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
```

**To use:**
```bash
python3 train_my_custom_model.py --data data --epochs 20
```

---

## 📚 Quick Reference

### File Size Calculator for FC Layer

```python
# Formula: final_size = input_size / (2 ^ num_pooling_layers)

# Examples with 4 pooling layers (each divides by 2):
input_size = 224  → final_size = 224/16 = 14  → Linear(256 * 14 * 14, ...)
input_size = 128  → final_size = 128/16 = 8   → Linear(256 * 8 * 8, ...)
input_size = 512  → final_size = 512/16 = 32  → Linear(256 * 32 * 32, ...)

# For 3 pooling layers:
input_size = 224  → final_size = 224/8 = 28   → Linear(128 * 28 * 28, ...)
```

### Common Parameter Ranges

```python
# Hyperparameters:
BATCH_SIZE: 8, 16, 32, 64  (smaller for larger images)
LEARNING_RATE: 0.0001, 0.001, 0.01  (smaller is safer)
EPOCHS: 10-30  (transfer learning needs fewer)
IMG_SIZE: 64, 128, 224, 512  (larger = better but slower)

# Architecture:
Conv filters: 32, 64, 128, 256, 512  (powers of 2)
FC hidden units: 128, 256, 512, 1024  (powers of 2)
Dropout: 0.3, 0.5  (0.5 is standard)
```

---

## ✅ Checklist Before Modifying Code

- [ ] Read the original code completely
- [ ] Understand what each line does
- [ ] Make a backup copy of the file
- [ ] Test your changes on a tiny dataset first
- [ ] Check if you can achieve the same result with command-line flags
- [ ] Document what you changed and why
- [ ] Verify the model architecture makes sense (use `print(model)`)

---

## 🆘 When Something Breaks

### Common Issues After Modification:

**Error: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`**
→ FC layer input size wrong - recalculate using formula above

**Error: `RuntimeError: Given groups=1, expected weight to be...`**
→ Conv layer channels don't match - check in_channels = prev out_channels

**Error: Loss is NaN after changes**
→ Learning rate might be wrong for new architecture - try 10x smaller

**Model trains but accuracy is worse**
→ You might have broken something - revert and try smaller changes

---

## 📖 Further Reading

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Transfer Learning Tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- Custom Datasets: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- Model Debugging: https://karpathy.github.io/2019/04/25/recipe/

---

**Questions? Start with small modifications and test thoroughly!** 🚀
