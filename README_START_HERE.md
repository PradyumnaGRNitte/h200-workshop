
# 🚀 START HERE - H200 GPU Cluster Quick Start Guide

**First time using H200 for your own project? Read this first!**

---

## 📋 Table of Contents
1. [Which Script Should I Use?](#which-script-should-i-use)
2. [Quick Start for Your Custom Data](#quick-start-for-your-custom-data)
3. [Common Problems & Solutions](#common-problems--solutions)
4. [Understanding Results](#understanding-results)
5. [When to Ask for Help](#when-to-ask-for-help)

---

## 🤔 Which Script Should I Use?

### Decision Tree:

```
START HERE
    ↓
Do you have < 500 total images?
    ├─ YES → Use train_transfer.py (ResNet-18 Transfer Learning)
    │        It's pre-trained on 1.2M images!
    │
    └─ NO → Continue to next question
        ↓
Are your images natural photos or medical images?
    ├─ YES → Use train_transfer.py
    │        Pre-trained features work great!
    │
    └─ NO → Continue to next question
        ↓
Do you have > 5000 images?
    ├─ YES → Try BOTH train_cnn.py AND train_transfer.py
    │        Compare which works better
    │
    └─ NO → Use train_transfer.py
            Safer choice for medium datasets
```

### Quick Reference Table:

| Your Situation | Use This Script | Why? |
|----------------|----------------|------|
| **< 100 images total** | ⚠️ STOP! Get more data first | Not enough to train any model |
| **100-500 images** | `train_transfer.py` | Pre-trained features help with small data |
| **Medical images (X-rays, CT, MRI)** | `train_transfer.py` | ImageNet features transfer well |
| **Natural photos (plants, animals, objects)** | `train_transfer.py` | Perfect match for pre-training |
| **Textures, patterns, specialized domains** | `train_cnn.py` | Custom features needed |
| **> 5000 images, ample compute time** | `train_cnn.py` or `train_transfer.py` | Both should work, try both! |
| **Simple patterns (< 64x64)** | `train_nn_mnist.py` | Fastest to test |

---

## 🏃 Quick Start for Your Custom Data

### Step 1: Organize Your Data

**CRITICAL: Your folder structure must be EXACTLY like this:**

```
data/
├── train/
│   ├── class1/           ← Put 80% of class1 images here
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── class2/           ← Put 80% of class2 images here
│   │   ├── img001.jpg
│   │   └── ...
│   └── class3/           ← And so on...
│       └── ...
└── test/
    ├── class1/           ← Put 20% of class1 images here
    │   └── ...
    ├── class2/           ← Put 20% of class2 images here
    │   └── ...
    └── class3/
        └── ...
```

**⚠️ COMMON MISTAKES:**
- ❌ Putting all images in one folder
- ❌ Having a folder called "data/class1" instead of "data/train/class1"
- ❌ Using different folder names than your actual classes
- ❌ Forgetting to split train/test

### Step 2: Upload to H200

```bash
# From your laptop/workstation:
scp -r data/ your_username@172.18.0.83:~/my_project/
```

### Step 3: Decide Which Script to Use

**Follow the decision tree above, then:**

```bash
# SSH to H200
ssh your_username@172.18.0.83

# Activate environment
cd ~/my_project
source ~/env/bin/activate

# For transfer learning (RECOMMENDED for most cases):
python3 train_transfer.py --data data --epochs 15 --batch-size 32

# For custom CNN from scratch:
python3 train_cnn.py --data data --epochs 25 --batch-size 32

# For simple NN (only if images are small, < 64x64):
python3 train_nn_mnist.py --data data
```

### Step 4: Monitor Training

```bash
# Watch it train (live updates)
tail -f logs/*.out

# Or check job status
squeue --me

# Press Ctrl+C to stop watching (training continues!)
```

### Step 5: Check Results

```bash
# After training completes:
cat results/transfer_training_metrics.txt
# or
cat results/cnn_training_metrics.txt

# View your model
ls -lh models/
```

---

## 🔧 Common Problems & Solutions

### Problem 1: Accuracy Stuck at 1/num_classes

**Example:** 3 classes, accuracy stuck at 33% after 10 epochs

**What this means:** Model is guessing randomly - something is fundamentally wrong

**FIX:**
```bash
# 1. Check your folder structure
ls -R data/

# Should see:
# data/train/class1/
# data/train/class2/
# data/train/class3/
# data/test/class1/
# data/test/class2/
# data/test/class3/

# 2. Check you have images in each folder
ls data/train/class1/ | wc -l  # Should NOT be 0

# 3. Check folder names match your classes exactly
```

**If structure is correct:**
- Labels might be corrupted
- Images might all be identical
- Data might not have loaded (check file permissions)

---

### Problem 2: CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**FIX:**
```bash
# Reduce batch size (try in this order):
python3 train_transfer.py --batch-size 16   # Try first
python3 train_transfer.py --batch-size 8    # If still fails
python3 train_transfer.py --batch-size 4    # Last resort

# OR reduce image size:
python3 train_transfer.py --img-size 128    # Down from 224

# OR both:
python3 train_transfer.py --batch-size 8 --img-size 128
```

---

### Problem 3: Training is Too Slow

**Symptoms:** 1 epoch takes > 30 minutes

**FIX:**
```bash
# 1. Reduce image size
python3 train_transfer.py --img-size 128  # Faster!

# 2. Reduce epochs (you don't always need 20+)
python3 train_transfer.py --epochs 10

# 3. Check you're actually using GPU
python3 verify_setup.py  # Should say "CUDA available"
```

---

### Problem 4: Loss is NaN

**Error:** Loss shows as `nan` after a few iterations

**CAUSES:**
- Learning rate too high
- Data normalization wrong
- Corrupted images

**FIX:**
```bash
# Reduce learning rate
python3 train_transfer.py --lr 0.0001   # Down from 0.001

# Or try even lower
python3 train_transfer.py --lr 0.00001
```

---

### Problem 5: Train Accuracy 99%, Test Accuracy 60%

**What this means:** Overfitting - model memorized training data

**FIX:**
1. **Get more training data** (best solution)
2. **Use transfer learning** (if not already)
   ```bash
   python3 train_transfer.py --data data
   ```
3. **More data augmentation** (only for custom CNN)
4. **Reduce model complexity** (fewer layers)

---

### Problem 6: Loss Doesn't Decrease

**Symptoms:** Loss stays flat or increases

**FIXES:**
```bash
# 1. Check learning rate
python3 train_transfer.py --lr 0.0001  # Try smaller

# 2. Check data loading
ls data/train/class1/ | head -5  # Should show image files

# 3. Try different script
# If using custom CNN, try transfer learning:
python3 train_transfer.py --data data

# 4. Check labels
# Make sure folder names match class names exactly
```

---

### Problem 7: FileNotFoundError

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'data/train'`

**FIX:**
```bash
# Check where you are
pwd  # Should be in ~/my_project

# Check data exists
ls -l data/
ls -l data/train/
ls -l data/test/

# If missing, upload again:
# From your laptop: scp -r data/ user@172.18.0.83:~/my_project/
```

---

## 📊 Understanding Results

### What's a Good Accuracy?

| Dataset Size | Method | Expected Accuracy | Time |
|--------------|--------|-------------------|------|
| 100-500 images (2 classes) | Transfer Learning | **70-85%** | 5-10 min |
| 500-2000 images (3-5 classes) | Transfer Learning | **75-90%** | 10-20 min |
| 2000-5000 images (5-10 classes) | Transfer Learning | **80-92%** | 15-30 min |
| 5000+ images (10+ classes) | Custom CNN | **85-95%** | 30-60 min |

**⚠️ If your accuracy is much lower than these ranges → something is wrong!**

### Red Flags:

🚩 **Accuracy = exactly 1/num_classes** (e.g., 33% for 3 classes)
   → Model is guessing randomly - check your data/labels!

🚩 **Train 99%, Test < 70%**
   → Overfitting - need more data or use transfer learning

🚩 **Accuracy < 50% after 10 epochs**
   → Something fundamentally wrong - check data, labels, model choice

🚩 **Loss is NaN**
   → Learning rate too high or data problem

### Green Flags:

✅ **Loss steadily decreasing**
✅ **Train and test accuracy close (within 10%)**
✅ **Accuracy > 75% for your dataset size**
✅ **Training completes without errors**

---

## ❓ When to Ask for Help

### DON'T Ask Yet If:
- You haven't tried the fixes above
- You're on epoch 1 or 2 (wait for at least 5 epochs)
- You haven't checked your folder structure
- You haven't tried different batch sizes

### DO Ask If:
- ✅ Accuracy < 50% after 10 epochs
- ✅ Loss is NaN or doesn't decrease at all
- ✅ You've tried 3 different settings and nothing works
- ✅ You get an error you don't understand
- ✅ Training takes > 2 hours for < 1000 images

### How to Ask for Help:

**Include this information:**
1. What script you're using (`train_transfer.py` or `train_cnn.py`)
2. Your dataset size (how many images per class)
3. The exact error message (copy-paste)
4. What you've already tried
5. Output of `ls -R data/` (folder structure)

**Contact:**
📧 Email: pradyumna@nitte.edu.in
💬 Subject: "H200 Help: [Brief Description]"

---

## 🎓 Learning Resources

### Before You Start:
- Understand basic neural networks (Coursera Deep Learning)
- Know your data (what are you classifying?)
- Have realistic expectations (< 100 images = tough)

### While Training:
- Watch the loss decrease (good sign!)
- Monitor train vs test accuracy (should be close)
- Be patient (good models take time)

### After Training:
- Test on new images (not in train/test sets)
- Visualize predictions (see where it fails)
- Iterate (more data, better preprocessing)

---

## 🚀 Quick Examples

### Example 1: Medical X-Ray Classification (2 classes)

```bash
# You have 400 normal + 400 pneumonia X-rays
# Already organized in data/train and data/test

cd ~/my_project
source ~/env/bin/activate

# Use transfer learning (best for medical images)
python3 train_transfer.py \
    --data data \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.001

# Expected: 80-90% accuracy in 10-15 minutes
```

### Example 2: Plant Disease Classification (5 classes)

```bash
# You have 500 images per disease (2500 total)

cd ~/my_project
source ~/env/bin/activate

# Transfer learning works great for natural images
python3 train_transfer.py \
    --data data \
    --epochs 20 \
    --batch-size 32 \
    --img-size 224

# Expected: 85-92% accuracy in 20-30 minutes
```

### Example 3: Custom Texture Classification (10 classes)

```bash
# You have 1000 images per texture (10000 total)
# Textures might need custom features

cd ~/my_project
source ~/env/bin/activate

# Try custom CNN first
python3 train_cnn.py \
    --data data \
    --epochs 30 \
    --batch-size 64 \
    --img-size 224

# Compare with transfer learning
python3 train_transfer.py \
    --data data \
    --epochs 20 \
    --batch-size 64

# Use whichever gives better results
```

---

## ✅ Final Checklist Before Training

- [ ] Data organized in `data/train/classN/` and `data/test/classN/`
- [ ] At least 50 images per class (100+ recommended)
- [ ] Roughly balanced classes (similar number of images)
- [ ] Images are readable (not corrupted)
- [ ] Decided which script to use (see decision tree)
- [ ] Activated virtual environment (`source ~/env/bin/activate`)
- [ ] Know how to monitor training (`tail -f logs/*.out`)

---

**Now you're ready! Good luck! 🎉**

Questions? Read "Common Problems & Solutions" above or contact pradyumna@nitte.edu.in
