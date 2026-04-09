# Workshop Day Setup Instructions

## For Participants (Workshop Day)

### Quick Setup (5 minutes)

```bash
# 1. SSH to H200 server
ssh your_username@172.18.0.83

# 2. Clone repository
git clone https://github.com/YOUR_USERNAME/h200-workshop.git
cd h200-workshop

# 3. Run automated setup
bash setup.sh

# Done! You're ready to start training.
```

---

## What the Setup Script Does

1. ✅ Creates Python virtual environment at `~/env`
2. ✅ Installs PyTorch with CUDA 12.1 support
3. ✅ Installs all dependencies (matplotlib, pillow, numpy)
4. ✅ Creates project directory at `~/my_project`
5. ✅ Copies all workshop files to `~/my_project`
6. ✅ Pre-downloads MNIST dataset (saves time!)
7. ✅ Runs verification to ensure everything works

---

## Manual Setup (If Automated Fails)

### Step 1: Create Virtual Environment
```bash
python3 -m venv ~/env
source ~/env/bin/activate
```

### Step 2: Install PyTorch with CUDA
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Dependencies
```bash
pip install matplotlib pillow numpy
```

### Step 4: Create Project Structure
```bash
mkdir -p ~/my_project/{data,models,results,logs}
cp *.py ~/my_project/
cp *.sh ~/my_project/
cd ~/my_project
```

### Step 5: Verify Setup
```bash
python3 verify_setup.py
```

---

## Running the Demo

### Start Training
```bash
cd ~/my_project
source ~/env/bin/activate
sbatch launch_train.sh
```

### Monitor Progress
```bash
# Check job status
squeue --me

# Watch training output
tail -f logs/mnist_*.out
```

### View Results
```bash
# Training metrics
cat results/training_metrics.txt

# Saved model
ls -lh models/mnist_nn.pth
```

---

## Troubleshooting

### "ModuleNotFoundError: torch"
```bash
source ~/env/bin/activate
```

### "Permission denied: setup.sh"
```bash
chmod +x setup.sh
bash setup.sh
```

### "Job pending (PD)"
```bash
# Check queue
squeue --all

# Wait a moment, GPU will be available soon
# Average wait time < 5 minutes
```

### Setup script fails
```bash
# Try manual setup above
# OR contact: pradyumna@nitte.edu.in
```

---

## Expected Timeline

- **Setup:** 5 minutes
- **First training run:** 2 minutes
- **Total:** ~7 minutes from clone to results

---

## For Instructors

### Pre-Workshop Preparation

1. **Test the repository:**
   ```bash
   git clone <repo-url>
   cd h200-workshop
   bash setup.sh
   sbatch launch_train.sh
   ```

2. **Verify all files:**
   ```bash
   python3 verify_setup.py
   ls -la ~/my_project/
   ```

3. **Pre-download datasets** (optional, saves time):
   ```bash
   cd ~/my_project
   source ~/env/bin/activate
   python3 -c "from torchvision import datasets; datasets.MNIST('./data', download=True)"
   ```

4. **Have backup ready:**
   - Screenshots of successful training
   - Video recording of demo
   - Printed copies of WORKSHOP_DEMO_SCRIPT.md

---

## Repository Structure

```
h200-workshop/
├── README.md                    # Main documentation
├── SETUP_INSTRUCTIONS.md        # This file
├── QUICKREF.md                  # Quick command reference
├── WORKSHOP_DEMO_SCRIPT.md     # Complete workshop guide
├── CONTRIBUTING.md              # Contribution guidelines
│
├── setup.sh                     # Automated setup
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── Training Scripts:
│   ├── train_nn_mnist.py        # MNIST demo (fastest)
│   ├── train_cnn.py             # Custom CNN
│   └── train_transfer.py        # Transfer learning
│
├── Utilities:
│   ├── verify_setup.py          # Setup verification
│   ├── predict.py               # Inference
│   └── launch_train.sh          # Slurm script
│
└── example_structure/           # Directory layout example
    ├── data/
    ├── models/
    ├── results/
    └── logs/
```

---

## Success Criteria

After setup, you should have:
- ✅ Virtual environment at `~/env`
- ✅ PyTorch with CUDA support installed
- ✅ Project directory at `~/my_project`
- ✅ All workshop files copied
- ✅ MNIST dataset downloaded
- ✅ `verify_setup.py` passes all checks

---

## Support

**Email:** pradyumna@nitte.edu.in

**Include:**
- Your username
- Department
- Error message (full output)
- What you were trying to do

---

## Quick Reference

```bash
# Activate environment (every login)
source ~/env/bin/activate

# Navigate to project
cd ~/my_project

# Submit training job
sbatch launch_train.sh

# Check status
squeue --me

# Watch output
tail -f logs/mnist_*.out

# Cancel job
scancel <job_id>
```

---

**Ready to start? Run `bash setup.sh` and you're good to go!** 🚀
