"""
MNIST Model Prediction Script
H200 GPU Cluster Workshop Demo

Loads a trained model and makes predictions on new digit images.
Outputs match the format shown in the workshop slides.
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt


# Define the same neural network architecture
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
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def preprocess_image(image_path):
    """
    Preprocess an image for MNIST prediction.
    Converts to grayscale, resizes to 28x28, normalizes.
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32)
    
    # Invert if needed (MNIST has white digits on black background)
    # Check if image is mostly white (assume black digit on white background)
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Apply MNIST normalization (mean=0.1307, std=0.3081)
    img_array = (img_array - 0.1307) / 0.3081
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor, img_array


def predict(model, image_tensor, device):
    """
    Make a prediction on a single image.
    Returns predicted class and confidence.
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item() * 100


def visualize_prediction(image_array, predicted_class, confidence, output_path):
    """
    Create a visualization of the prediction.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image_array, cmap='gray')
    plt.title(f'Predicted: {predicted_class} (Confidence: {confidence:.1f}%)', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Predict digit from image using trained MNIST model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/mnist_nn.pth', 
                        help='Path to trained model (default: models/mnist_nn.pth)')
    parser.add_argument('--output', type=str, default='results/prediction_output.png',
                        help='Path to save visualization (default: results/prediction_output.png)')
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first using train_nn_mnist.py")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model}")
    model = SimpleNN().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model trained for {checkpoint['epoch']} epochs")
    print(f"Training accuracy: {checkpoint['train_acc']:.2f}%")
    print(f"Test accuracy: {checkpoint['test_acc']:.2f}%")
    print("-" * 60)
    
    # Preprocess image
    print(f"Processing image: {args.image}")
    image_tensor, image_array = preprocess_image(args.image)
    
    # Make prediction
    predicted_class, confidence = predict(model, image_tensor, device)
    
    # Output results (matching workshop slide format)
    print(f"\nPrediction : {predicted_class}")
    print(f"Confidence : {confidence:.1f}%")
    
    # Create visualization
    visualize_prediction(image_array, predicted_class, confidence, args.output)
    print(f"Saved to   : {args.output}")
    print("-" * 60)


if __name__ == '__main__':
    main()
