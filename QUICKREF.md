# H200 GPU Cluster - Quick Reference

## 🔐 Login

```bash
ssh your_username@172.18.0.83
passwd  # Change password on first login
```

## 📦 Environment Setup (First Time Only)

```bash
# Create directories
mkdir -p ~/my_project/{data,models,results,logs}

# Create virtual environment
python3 -m venv ~/env

# Activate environment
source ~/env/bin/activate

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib pillow numpy
```

## 🔄 Every Time You Login

```bash
source ~/env/bin/activate
cd ~/my_project
```

## 🎯 Quick Commands

### Check Resources
```bash
list-gpus                    # See available GPU slices
squeue --all                 # See all jobs
squeue --me                  # See your jobs
sacctmgr show association user=$(whoami) format=User,GrpTRES,QOS  # Your limits
```

### Submit Jobs
```bash
sbatch launch_train.sh       # Submit batch job (recommended)
srun --partition=h200 --gres=gpu:1g.18gb:1 --pty bash  # Interactive session
```

### Monitor Jobs
```bash
tail -f logs/out.txt         # Watch live output
squeue --me                  # Check job status
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS  # Job statistics
```

### Cancel Jobs
```bash
scancel <job_id>             # Cancel specific job
scancel --user=$(whoami)     # Cancel all your jobs
```

### Storage
```bash
du -sh ~/                    # Check total usage
du -sh ~/*/                  # Check per-directory usage
rm -rf ~/my_project/data/    # Delete dataset (important!)
```

## 🚀 Training Examples

### MNIST Demo (Fastest)
```bash
sbatch launch_train.sh
tail -f logs/out.txt
```

### Custom Dataset (CNN)
```bash
# Upload data first, then:
srun --partition=h200 --gres=gpu:1g.18gb:1 --pty bash
source ~/env/bin/activate
cd ~/my_project
python3 train_cnn.py --data data --epochs 20
exit  # ALWAYS exit when done!
```

### Transfer Learning (Best Accuracy)
```bash
srun --partition=h200 --gres=gpu:1g.18gb:1 --pty bash
source ~/env/bin/activate
cd ~/my_project
python3 train_transfer.py --data data --epochs 10
exit
```

## 📥 Download Results

```bash
# On your LOCAL machine:
scp user@172.18.0.83:~/my_project/models/mnist_nn.pth ./
scp user@172.18.0.83:~/my_project/results/* ./results/
```

## 🐛 Common Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: torch` | `source ~/env/bin/activate` |
| Job stuck in PD (AssocGrpGRES) | `exit` your srun session first |
| CUDA out of memory | Reduce batch size: `BATCH_SIZE = 16` |
| Invalid generic resource | Check your slice: `sacctmgr show association` |

## ✅ Best Practices

- ✓ ALWAYS use Slurm (sbatch or srun), never run on login node
- ✓ Check `list-gpus` before submitting
- ✓ Exit srun immediately when done
- ✓ Delete data after training: `rm -rf ~/my_project/data/`
- ✓ Stay under 50 GB storage limit
- ✓ Use batch jobs (sbatch) for long runs

## 🔍 Job States

| State | Meaning |
|-------|---------|
| PD | Pending - waiting for GPU |
| R | Running - actively training |
| CG | Completing - finishing up |

## 📧 Support

**Contact:** pradyumna@nitte.edu.in

Include: username, department, error message, what you were doing
