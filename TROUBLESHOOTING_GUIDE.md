# 🔧 H200 GPU Cluster - Complete Troubleshooting Guide

**Comprehensive guide to fixing common issues when training your own models**

---

## 📋 Table of Contents

1. [Data & Loading Errors](#1-data--loading-errors)
2. [Training Issues](#2-training-issues)
3. [Memory & Resource Errors](#3-memory--resource-errors)
4. [Model Performance Problems](#4-model-performance-problems)
5. [Slurm & Job Errors](#5-slurm--job-errors)
6. [Environment & Setup Issues](#6-environment--setup-issues)
7. [Quick Reference Table](#7-quick-reference-table)

---

## 1. Data & Loading Errors

### Error 1.1: FileNotFoundError

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/train'
```

**CAUSE:** Script can't find your data folder

**DIAGNOSIS:**
```bash
# Check where you are
pwd

# Should be in ~/my_project
# If not: cd ~/my_project

# Check data exists
ls -l data/
ls -l data/train/
ls -l data/test/
```

**FIX:**
```bash
# If missing, upload from your laptop:
scp -r ./my_data/ user@172.18.0.83:~/my_project/data/

# Make sure you're in the right directory
cd ~/my_project

# Try again
source ~/env/bin/activate
python3 train_transfer.py --data data
```

---

### Error 1.2: RuntimeError - Found 0 files

```
RuntimeError: Found 0 files in subfolders of: data/train
```

**CAUSE:** Wrong folder structure or empty folders

**DIAGNOSIS:**
```bash
# Check folder structure
ls -R data/

# Should see:
# data/train/class1/image1.jpg
# data/train/class2/image1.jpg
# data/test/class1/image1.jpg
# data/test/class2/image1.jpg

# Count images per class
ls data/train/class1/ | wc -l  # Should NOT be 0
ls data/train/class2/ | wc -l
```

**COMMON MISTAKES:**
```bash
# ❌ WRONG:
data/
  class1/image1.jpg
  class2/image1.jpg

# ✅ CORRECT:
data/
  train/
    class1/image1.jpg
    class2/image1.jpg
  test/
    class1/image1.jpg
    class2/image1.jpg
```

**FIX:**
```bash
# Reorganize your data with the correct structure
mkdir -p data/train data/test

# Move images to correct locations
# Example for 2 classes with 100 images each:
mv class1_images/*.jpg data/train/class1/  # First 80 images
mv class1_images_test/*.jpg data/test/class1/  # Last 20 images
# Repeat for class2...
```

---

### Error 1.3: OSError - Image file is truncated

```
OSError: image file is truncated (X bytes not processed)
```

**CAUSE:** Corrupted or incomplete image files

**DIAGNOSIS:**
```bash
# Find corrupted images
cd data/train/class1
for img in *.jpg; do
    identify "$img" > /dev/null 2>&1 || echo "Corrupted: $img"
done
```

**FIX:**
```bash
# Option 1: Remove corrupted images
rm corrupted_image.jpg

# Option 2: Re-download/re-upload corrupted files
scp ./fixed_image.jpg user@172.18.0.83:~/my_project/data/train/class1/
```

---

### Error 1.4: ValueError - Number of classes mismatch

```
ValueError: Expected input batch_size (32) to match target batch_size (16)
```

**CAUSE:** Inconsistent number of classes between train and test sets

**DIAGNOSIS:**
```bash
# Check train classes
ls data/train/

# Check test classes
ls data/test/

# Should be IDENTICAL
```

**FIX:**
```bash
# Make sure both have same classes
# If train has: class1, class2, class3
# Then test MUST have: class1, class2, class3

# Create missing test folders
mkdir -p data/test/class3
# Add some test images
```

---

## 2. Training Issues

### Error 2.1: Loss is NaN

```
Epoch [1/10]  Loss: nan  Acc: 0.0%
```

**CAUSE:** Learning rate too high, data normalization wrong, or gradient explosion

**DIAGNOSIS:**
```bash
# Check what learning rate you're using
# Default is usually 0.001
```

**FIX:**
```bash
# Try much lower learning rate
python3 train_transfer.py --lr 0.0001  # 10x smaller

# Or even lower
python3 train_transfer.py --lr 0.00001  # 100x smaller

# If still NaN, check data:
# - Are images corrupted?
# - Are labels correct?
# - Try different script (transfer vs CNN)
```

---

### Error 2.2: Loss Doesn't Decrease

```
Epoch [1/10]  Loss: 2.3026  Acc: 10.0%
Epoch [2/10]  Loss: 2.3025  Acc: 10.1%
Epoch [3/10]  Loss: 2.3024  Acc: 10.2%
...stays flat...
```

**CAUSE:** Learning rate too low, wrong labels, or model can't learn from data

**DIAGNOSIS:**
```bash
# Check if loss is EXACTLY log(num_classes)
# For 10 classes: log(10) = 2.3026
# This means random guessing!

# Check labels
ls data/train/
# Folder names should match your actual classes
```

**FIX:**

**Fix 1: Try higher learning rate**
```bash
python3 train_transfer.py --lr 0.01  # 10x higher than default
```

**Fix 2: Check data/labels**
```bash
# Make sure:
# 1. Folder names = class names
# 2. Images are in correct folders
# 3. Images aren't all identical

# View a random image to verify
ls data/train/class1/ | head -1
# Copy to results and check it
cp data/train/class1/[filename] results/check.jpg
```

**Fix 3: Try different model**
```bash
# If using custom CNN, try transfer learning
python3 train_transfer.py --data data

# Or vice versa
python3 train_cnn.py --data data
```

---

### Error 2.3: Accuracy Stuck at 1/num_classes

```
Epoch [5/10]  Loss: 1.0986  Test Acc: 33.3%  # For 3 classes
Epoch [10/10]  Loss: 1.0985  Test Acc: 33.4%  # Still ~33%
```

**CAUSE:** Model is guessing randomly - labels are wrong!

**DIAGNOSIS:**
```bash
# For N classes, accuracy should NOT be close to 100/N
# 2 classes: shouldn't stay at ~50%
# 3 classes: shouldn't stay at ~33%
# 10 classes: shouldn't stay at ~10%

# This means: LABELS ARE WRONG or DATA LOADING FAILED
```

**FIX:**

**Step 1: Verify folder structure**
```bash
ls -R data/

# Must see:
# data/train/classA/
# data/train/classB/
# data/test/classA/
# data/test/classB/
```

**Step 2: Verify images exist**
```bash
# Each folder should have images
ls data/train/classA/ | head -5
ls data/train/classB/ | head -5

# Check counts
ls data/train/classA/ | wc -l  # Should be > 0
ls data/train/classB/ | wc -l  # Should be > 0
```

**Step 3: Check folder names**
```bash
# Folder names must be exactly your classes
# ❌ WRONG: class_1, Class1, CLASS1
# ✅ CORRECT: consistent naming (e.g., all lowercase)
```

**Step 4: If structure is correct, check image quality**
```bash
# Images might all be identical or corrupted
file data/train/classA/*.jpg | head

# Check file sizes
du -sh data/train/classA/*.jpg | sort -h | head
# If all identical size → might be duplicates
```

---

### Error 2.4: Training Accuracy High, Test Accuracy Low

```
Epoch [15/20]  Train Acc: 99.2%  Test Acc: 62.1%
```

**CAUSE:** Overfitting - model memorized training data

**DIAGNOSIS:**
```bash
# Gap > 20% is serious overfitting
# Gap > 30% is severe overfitting
```

**FIX:**

**Best Fix: Use transfer learning (if not already)**
```bash
python3 train_transfer.py --data data --epochs 15
```

**Alternative Fixes:**

**1. Get more data** (best long-term solution)
- Need at least 100 images per class
- 500+ per class is much better

**2. More data augmentation** (for custom CNN only)
- Already included in train_cnn.py
- Edit to add more: rotation, flipping, color jittering

**3. Early stopping**
```bash
# Stop training when test accuracy stops improving
# Watch the output - usually peaks around epoch 10-15
# Kill training with Ctrl+C when you see test accuracy drop
```

**4. Reduce model complexity**
- Use transfer learning (fewer trainable parameters)
```bash
python3 train_transfer.py --freeze-layers 7  # Default
python3 train_transfer.py --freeze-layers 8  # Even more frozen
```

---

## 3. Memory & Resource Errors

### Error 3.1: CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate 1.95 GiB 
(GPU 0; 17.12 GiB total capacity; 15.89 GiB already allocated)
```

**CAUSE:** Batch size or image size too large for GPU slice

**DIAGNOSIS:**
```bash
# Check your GPU slice
squeue --me
# Look at GRES column: 1g.18gb or 2g.35gb
```

**FIX (try in order):**

**Fix 1: Reduce batch size**
```bash
# Default is usually 32
python3 train_transfer.py --batch-size 16  # Try first
python3 train_transfer.py --batch-size 8   # If still fails
python3 train_transfer.py --batch-size 4   # Last resort
```

**Fix 2: Reduce image size**
```bash
python3 train_transfer.py --img-size 128  # Down from 224
python3 train_transfer.py --img-size 96   # Even smaller
```

**Fix 3: Both**
```bash
python3 train_transfer.py --batch-size 8 --img-size 128
```

**Fix 4: Request larger GPU slice**
```bash
# Edit launch_train.sh
# Change from:
#SBATCH --gres=gpu:1g.18gb:1
# To:
#SBATCH --gres=gpu:2g.35gb:1

# Resubmit
sbatch launch_train.sh
```

---

### Error 3.2: Training Takes Forever

**SYMPTOM:** 1 epoch takes > 30 minutes

**DIAGNOSIS:**
```bash
# Check if actually using GPU
nvidia-smi

# Should show python process using GPU
```

**FIX:**

**1. Verify GPU usage**
```bash
python3 verify_setup.py
# Should say "CUDA available: True"
```

**2. Reduce image size**
```bash
python3 train_transfer.py --img-size 128  # Faster than 224
```

**3. Reduce epochs**
```bash
# You don't always need 20+ epochs
python3 train_transfer.py --epochs 10
```

**4. Use transfer learning (if not already)**
```bash
# Transfer learning is MUCH faster than CNN from scratch
python3 train_transfer.py --data data --epochs 10
# vs
python3 train_cnn.py --data data --epochs 30  # Slower!
```

---

## 4. Model Performance Problems

### Error 4.1: Accuracy Lower Than Expected

**SYMPTOM:** Getting 60% when you expected 80-90%

**DIAGNOSIS CHECKLIST:**

```bash
# 1. Check dataset size
ls data/train/class1/ | wc -l
ls data/train/class2/ | wc -l

# Need at least 100 per class
# 500+ is much better

# 2. Check class balance
# If class1 has 1000 images and class2 has 50
# Model will be biased!

# 3. Check data quality
ls -lh data/train/class1/ | head
# All files should have reasonable sizes
# Not all identical (duplicates)

# 4. Check if using right script
# Small data (< 500 imgs) → MUST use transfer learning
```

**FIX:**

**1. Use transfer learning**
```bash
python3 train_transfer.py --data data --epochs 15
```

**2. More epochs**
```bash
python3 train_transfer.py --epochs 20  # Up from 10
```

**3. Better hyperparameters**
```bash
# Try different learning rate
python3 train_transfer.py --lr 0.0001  # Smaller
python3 train_transfer.py --lr 0.01    # Larger
```

**4. Get more data**
- Collect more samples
- Balance classes
- Check image quality

---

### Error 4.2: Model Works on Training Data but Fails on New Images

**SYMPTOM:** 95% on test set, but 30% on your new images

**CAUSE:** Data distribution mismatch

**DIAGNOSIS:**
```bash
# Your new images might be:
# - Different format (JPEG vs PNG)
# - Different size/resolution
# - Different lighting/contrast
# - Different camera/scanner
```

**FIX:**

**1. Check preprocessing**
```bash
# New images should match training:
# - Same normalization
# - Same resize method
# - Same color space (RGB vs grayscale)
```

**2. Add more diverse training data**
- Include samples from the same source as your new images
- Add data augmentation
- Collect data in different conditions

**3. Retrain with combined dataset**
```bash
# Add some of your "new" images to training set
# Retrain the model
```

---

## 5. Slurm & Job Errors

### Error 5.1: Job Stuck in Pending (PD)

```
JOBID  PARTITION  NAME      USER  ST  TIME
  123  h200       my_train  user  PD  0:00
```

**CAUSE:** GPU already allocated or queue is busy

**DIAGNOSIS:**
```bash
# Check job details
squeue --me --long

# Look at REASON column:
# - "AssocGrpGRES" → You're already using GPU
# - "Resources" → Queue is busy
# - "Priority" → Someone else has higher priority
```

**FIX:**

**For "AssocGrpGRES":**
```bash
# Check if you have an srun session open
squeue --me

# If you see an srun job running, exit it:
exit  # From srun shell

# Or cancel it:
scancel <job_id>

# Then resubmit
sbatch launch_train.sh
```

**For "Resources":**
```bash
# Queue is busy, just wait
# Or try smaller GPU slice:

# Edit launch_train.sh
# Change to 1g.18gb if you were using 2g.35gb
```

---

### Error 5.2: Job Failed Immediately

```
JOBID  PARTITION  NAME      USER  ST  TIME
  123  h200       my_train  user  F   0:01
```

**DIAGNOSIS:**
```bash
# Check the log file
cat logs/my_train_*.out

# Look for error messages at the end
tail -20 logs/my_train_*.out
```

**COMMON CAUSES:**
- Wrong path in Slurm script
- Environment not activated
- Python script has errors
- Data not found

**FIX:**
```bash
# Test the command manually first
cd ~/my_project
source ~/env/bin/activate
python3 train_transfer.py --data data --epochs 1

# If this works manually, check your launch_train.sh
# Make sure it has:
# - source ~/env/bin/activate
# - cd ~/my_project
# - Correct python command
```

---

## 6. Environment & Setup Issues

### Error 6.1: ModuleNotFoundError: torch

```
ModuleNotFoundError: No module named 'torch'
```

**CAUSE:** Virtual environment not activated

**FIX:**
```bash
# Activate environment
source ~/env/bin/activate

# You should see (env) in your prompt:
(env) user@h200-server:~/my_project$

# Try again
python3 train_transfer.py --data data
```

---

### Error 6.2: Permission Denied

```
bash: ./setup.sh: Permission denied
```

**FIX:**
```bash
# Make executable
chmod +x setup.sh

# Run again
bash setup.sh
```

---

## 7. Quick Reference Table

| Symptom | Most Likely Cause | Quick Fix |
|---------|------------------|-----------|
| **Accuracy = 33% (3 classes)** | Labels wrong | Check folder structure |
| **CUDA out of memory** | Batch size too large | `--batch-size 16` or `8` |
| **Loss is NaN** | Learning rate too high | `--lr 0.0001` |
| **Loss doesn't decrease** | Learning rate wrong or data issue | Try `--lr 0.01` or check labels |
| **Train 99%, Test 60%** | Overfitting | Use transfer learning, get more data |
| **FileNotFoundError** | Wrong directory or data missing | `cd ~/my_project`, check `data/` exists |
| **Job stuck PD** | Already using GPU | `exit` from srun, resubmit |
| **Training very slow** | Image size too large | `--img-size 128` |
| **ModuleNotFoundError** | Env not activated | `source ~/env/bin/activate` |
| **RuntimeError: Found 0 files** | Wrong folder structure | Check `data/train/class1/` format |

---

## 🆘 When All Else Fails

### Last Resort Checklist:

1. **Start fresh**
```bash
cd ~
rm -rf my_project
git clone https://github.com/PradyumnaGRNitte/h200-workshop.git
cd h200-workshop
bash setup.sh
# Try MNIST first to verify everything works
```

2. **Try the simplest possible case**
```bash
# Create tiny test dataset
mkdir -p test_data/train/{class1,class2}
mkdir -p test_data/test/{class1,class2}

# Copy just 10 images per class
# Train on this tiny set to isolate the problem
python3 train_transfer.py --data test_data --epochs 3
```

3. **Contact for help** with:
   - Exact error message (copy-paste)
   - Output of `ls -R data/`
   - What you've already tried
   - Output of `python3 verify_setup.py`
   
📧 pradyumna@nitte.edu.in

---

## 📚 Additional Resources

- **H200 User Guide:** `/path/to/user_guide.pdf`
- **PyTorch Documentation:** https://pytorch.org/docs/stable/index.html
- **Transfer Learning Guide:** https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- **GitHub Issues:** https://github.com/PradyumnaGRNitte/h200-workshop/issues

---

**Remember:** Most problems are data/label issues, not code issues! Always check your data first. 🔍
