#!/bin/bash
#SBATCH --job-name=mnist_train
#SBATCH --partition=h200
#SBATCH --gres=gpu:1g.18gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/my_project/logs/mnist_%j.out
#SBATCH --error=/home/%u/my_project/logs/mnist_%j.err

echo "=================================================="
echo "H200 GPU Cluster - MNIST Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=================================================="

# Create directories if they don't exist
mkdir -p /home/%u/my_project/{data,models,results,logs}

# Activate virtual environment
source /home/%u/env/bin/activate

# Navigate to project directory
cd /home/%u/my_project

# Run training script
python3 train_nn_mnist.py

echo "=================================================="
echo "Completed: $(date)"
echo "=================================================="
