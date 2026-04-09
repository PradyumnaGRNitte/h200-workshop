# H200 GPU Cluster Workshop - NMAMIT

Complete workshop materials for demonstrating GPU-accelerated deep learning on the H200 cluster.

## 🚀 Quick Start (Workshop Day)

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/h200-workshop.git
cd h200-workshop

# 2. Run setup script
bash setup.sh

# 3. Verify installation
source ~/env/bin/activate
python3 verify_setup.py

# 4. Run demo
sbatch launch_train.sh
```

That's it! Training starts immediately.

---

## 📁 Repository Contents

```
h200-workshop/
├── setup.sh                # Automated setup script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── QUICKREF.md           # Quick reference guide
│
├── Training Scripts:
│   ├── train_nn_mnist.py      # MNIST neural network (fastest demo)
│   ├── train_cnn.py           # Custom dataset CNN
│   └── train_transfer.py      # Transfer learning with ResNet-18
│
├── Utilities:
│   ├── verify_setup.py        # Environment verification
│   ├── predict.py            # Inference on trained models
│   └── launch_train.sh       # Slurm batch script
│
└── Workshop Materials:
    └── WORKSHOP_DEMO_SCRIPT.md  # Complete demo guide
```

---

## 🎯 Workshop Demo Scripts

### 1️⃣ MNIST Neural Network (Recommended for Demo)
**Time:** ~2 minutes on H200  
**Accuracy:** ~98%

```bash
sbatch launch_train.sh
tail -f logs/mnist_*.out
```

**Best for:** First-time users, quick demonstration

---

### 2️⃣ Custom Dataset CNN
**Time:** Varies by dataset size  
**Use case:** Your own image classification tasks

```bash
# Organize your data:
# data/train/class1/, data/train/class2/
# data/test/class1/, data/test/class2/

srun --partition=h200 --gres=gpu:1g.18gb:1 --pty bash
source ~/env/bin/activate
cd ~/my_project
python3 train_cnn.py --data data --epochs 20
exit
```

**Best for:** Custom research projects

---

### 3️⃣ Transfer Learning (ResNet-18)
**Time:** ~5-10 minutes  
**Accuracy:** Best accuracy with minimal training

```bash
srun --partition=h200 --gres=gpu:1g.18gb:1 --pty bash
source ~/env/bin/activate
cd ~/my_project
python3 train_transfer.py --data data --epochs 10
exit
```

**Best for:** Production-quality models, small datasets

---

## 🛠️ Manual Setup (If setup.sh fails)

### Step 1: Create Environment
```bash
python3 -m venv ~/env
source ~/env/bin/activate
```

### Step 2: Install PyTorch
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Create Project Structure
```bash
mkdir -p ~/my_project/{data,models,results,logs}
cp *.py ~/my_project/
cp *.sh ~/my_project/
cp *.md ~/my_project/
```

### Step 5: Verify
```bash
cd ~/my_project
python3 verify_setup.py
```

---

## 📊 Performance Benchmarks

| Task | CPU (16 cores) | H200 GPU | Speedup |
|------|----------------|----------|---------|
| MNIST (10 epochs) | ~25 min | ~1.5 min | 16.7x |
| Custom CNN (20 epochs) | ~2 hours | ~8 min | 15x |
| ResNet-18 Transfer | ~1 hour | ~5 min | 12x |

---

## 🎓 Use Cases

### Research
- Image classification
- Object detection
- Time series prediction
- NLP tasks

### Teaching
- Deep learning courses
- Student projects
- Lab assignments
- Thesis work

### Industry Collaboration
- Proof-of-concept
- Model prototyping
- Data analysis

---

## 📖 Documentation

- **README.md** (this file): Overview and quick start
- **QUICKREF.md**: Command reference
- **WORKSHOP_DEMO_SCRIPT.md**: Complete workshop guide

---

## 🆘 Common Issues

### ModuleNotFoundError: torch
```bash
source ~/env/bin/activate
# Then resubmit job
```

### Job stuck in PD (pending)
```bash
exit  # Close any srun sessions first
sbatch launch_train.sh
```

### CUDA out of memory
```python
# In training script, reduce batch size:
BATCH_SIZE = 16  # Instead of 32
```

---

## 📧 Support

**Cluster Admin:** pradyumna@nitte.edu.in

**Registration:** https://nmamit-hpc-registration.onrender.com

**Include in email:**
- Username
- Department
- Error message
- What you were trying to do

---

## 🎯 Quick Commands

```bash
# Check GPU availability
list-gpus

# Submit job
sbatch launch_train.sh

# Check job status
squeue --me

# Watch training
tail -f logs/mnist_*.out

# Cancel job
scancel <job_id>

# Check storage
du -sh ~/

# Delete dataset after training
rm -rf ~/my_project/data/
```

---

## 📝 Citation

If you use the H200 cluster for research, please acknowledge:

```
This research was supported by the H200 GPU cluster at CoE, NMAM Institute of Technology, Nitte (Deemed to be University).
```

---

## 🎉 Workshop Schedule

**Total Time:** 20 minutes

1. **Setup & Verification** (3 min) - `verify_setup.py`
2. **Submit Training** (2 min) - `sbatch launch_train.sh`
3. **Monitor Progress** (10 min) - Live training observation
4. **Results & Discussion** (3 min) - Review outputs
5. **Q&A** (2 min)

---

## 🔗 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Slurm Quick Start](https://slurm.schedmd.com/quickstart.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)

---

## 📜 License

MIT License - Free to use for research and education

---

**Happy GPU Training! 🚀**

For latest updates and issues: [GitHub Repository](https://github.com/YOUR_USERNAME/h200-workshop)
