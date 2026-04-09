#!/bin/bash

# H200 GPU Cluster Workshop - Automated Setup Script
# Run this script after cloning the repository

set -e  # Exit on error

echo "=================================================================="
echo "H200 GPU Cluster Workshop - Setup"
echo "=================================================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found: Python $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "$HOME/env" ]; then
    echo "   Virtual environment already exists at ~/env"
    read -p "   Overwrite? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ~/env
        python3 -m venv ~/env
        echo "   ✓ Created new virtual environment"
    else
        echo "   Using existing virtual environment"
    fi
else
    python3 -m venv ~/env
    echo "   ✓ Created virtual environment at ~/env"
fi

# Activate environment
echo ""
echo "🔧 Activating environment..."
source ~/env/bin/activate
echo "   ✓ Environment activated"

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet
echo "   ✓ pip upgraded"

# Install PyTorch with CUDA support
echo ""
echo "🔥 Installing PyTorch with CUDA 12.1 support..."
echo "   This may take a few minutes..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
echo "   ✓ PyTorch installed"

# Install other dependencies
echo ""
echo "📦 Installing additional dependencies..."
pip install -r requirements.txt --quiet
echo "   ✓ Dependencies installed"

# Create project directory structure
echo ""
echo "📁 Creating project directory structure..."
mkdir -p ~/my_project/{data,models,results,logs}
echo "   ✓ Created ~/my_project with subdirectories"

# Copy files to project directory
echo ""
echo "📄 Copying workshop files to ~/my_project..."
cp *.py ~/my_project/ 2>/dev/null || true
cp launch_train.sh ~/my_project/ 2>/dev/null || true
cp *.md ~/my_project/ 2>/dev/null || true
chmod +x ~/my_project/launch_train.sh
echo "   ✓ Files copied"

# Pre-download MNIST dataset
echo ""
echo "📥 Pre-downloading MNIST dataset..."
cd ~/my_project
python3 -c "from torchvision import datasets; datasets.MNIST('./data', train=True, download=True); datasets.MNIST('./data', train=False, download=True)" 2>/dev/null || echo "   ⚠️  Dataset download skipped (will download on first run)"
cd - > /dev/null

# Run verification
echo ""
echo "✅ Running setup verification..."
cd ~/my_project
python3 verify_setup.py
cd - > /dev/null

# Success message
echo ""
echo "=================================================================="
echo "✅ Setup Complete!"
echo "=================================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate environment:"
echo "   source ~/env/bin/activate"
echo ""
echo "2. Navigate to project:"
echo "   cd ~/my_project"
echo ""
echo "3. Run demo:"
echo "   sbatch launch_train.sh"
echo ""
echo "4. Monitor training:"
echo "   tail -f logs/mnist_*.out"
echo ""
echo "=================================================================="
echo ""
echo "For help: python3 verify_setup.py"
echo "Quick ref: cat QUICKREF.md"
echo ""
