#!/usr/bin/env python3
"""
H200 Cluster Setup Verification Script

Run this script to verify your environment is correctly configured
before submitting training jobs.

Usage:
    python3 verify_setup.py
"""

import sys
import os

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_check(name, passed, message=""):
    """Print a check result"""
    status = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {name}")
    if message:
        print(f"  → {message}")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    required_major, required_minor = 3, 8
    passed = version.major >= required_major and version.minor >= required_minor
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print_check(
        "Python Version",
        passed,
        f"Python {version_str} {'(OK)' if passed else f'(Require >= {required_major}.{required_minor})'}"
    )
    return passed

def check_torch():
    """Check if PyTorch is installed"""
    try:
        import torch
        print_check("PyTorch Installation", True, f"Version {torch.__version__}")
        return True
    except ImportError:
        print_check("PyTorch Installation", False, "Not installed - run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print_check("CUDA Availability", True, f"{device_name} (CUDA {cuda_version})")
        else:
            print_check("CUDA Availability", False, "GPU not accessible - make sure you're in a Slurm job (srun)")
        
        return cuda_available
    except:
        return False

def check_torchvision():
    """Check if torchvision is installed"""
    try:
        import torchvision
        print_check("torchvision", True, f"Version {torchvision.__version__}")
        return True
    except ImportError:
        print_check("torchvision", False, "Not installed - run: pip install torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False

def check_dependencies():
    """Check additional dependencies"""
    deps = {
        'PIL': ('Pillow', 'pip install pillow'),
        'numpy': ('NumPy', 'pip install numpy'),
        'matplotlib': ('Matplotlib', 'pip install matplotlib')
    }
    
    all_passed = True
    for module, (name, install_cmd) in deps.items():
        try:
            __import__(module)
            print_check(name, True)
        except ImportError:
            print_check(name, False, f"Not installed - run: {install_cmd}")
            all_passed = False
    
    return all_passed

def check_directories():
    """Check if project directories exist"""
    required_dirs = ['data', 'models', 'results', 'logs']
    all_exist = True
    
    for dir_name in required_dirs:
        exists = os.path.isdir(dir_name)
        if exists:
            print_check(f"Directory: {dir_name}/", True)
        else:
            print_check(f"Directory: {dir_name}/", False, f"Create with: mkdir {dir_name}")
            all_exist = False
    
    return all_exist

def check_slurm_env():
    """Check if running inside a Slurm job"""
    job_id = os.environ.get('SLURM_JOB_ID')
    
    if job_id:
        print_check("Slurm Environment", True, f"Running in job {job_id}")
        return True
    else:
        print_check("Slurm Environment", False, "Not in a Slurm job - this is OK for setup checks")
        print("  → When running training, use: sbatch launch_train.sh or srun")
        return False

def check_venv():
    """Check if running in virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print_check("Virtual Environment", True, f"Active: {sys.prefix}")
    else:
        print_check("Virtual Environment", False, "Not activated - run: source ~/env/bin/activate")
    
    return in_venv

def test_simple_model():
    """Test a simple PyTorch operation"""
    try:
        import torch
        
        # Create a small tensor and move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.randn(10, 10).to(device)
        y = torch.matmul(x, x)
        
        print_check("PyTorch Operation Test", True, "Matrix multiplication successful")
        return True
    except Exception as e:
        print_check("PyTorch Operation Test", False, f"Error: {str(e)}")
        return False

def main():
    """Run all verification checks"""
    print_header("H200 Cluster Setup Verification")
    
    print("\n📋 Checking Environment Configuration...")
    
    # Core checks
    python_ok = check_python_version()
    venv_ok = check_venv()
    torch_ok = check_torch()
    
    if not torch_ok:
        print("\n⚠️  PyTorch not installed. Please install it first:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)
    
    torchvision_ok = check_torchvision()
    cuda_ok = check_cuda()
    deps_ok = check_dependencies()
    
    print("\n📁 Checking Project Structure...")
    dirs_ok = check_directories()
    
    print("\n⚙️  Checking Runtime Environment...")
    slurm_ok = check_slurm_env()
    
    print("\n🧪 Testing PyTorch...")
    test_ok = test_simple_model()
    
    # Summary
    print_header("Verification Summary")
    
    critical_checks = [python_ok, torch_ok, torchvision_ok]
    recommended_checks = [venv_ok, deps_ok, dirs_ok]
    
    if all(critical_checks):
        print("✓ Critical components: OK")
    else:
        print("✗ Critical components: FAILED")
        print("  → Fix critical issues before proceeding")
    
    if all(recommended_checks):
        print("✓ Recommended setup: Complete")
    else:
        print("⚠️  Recommended setup: Incomplete")
        print("  → Some optional components missing")
    
    if cuda_ok:
        print("✓ GPU access: Available")
        print("\n🎉 Your setup is ready! You can now:")
        print("   1. Submit training jobs: sbatch launch_train.sh")
        print("   2. Run interactive sessions: srun --partition=h200 --gres=gpu:1g.18gb:1 --pty bash")
    else:
        print("⚠️  GPU access: Not available")
        print("  → This is normal if you're on the login node")
        print("  → GPU will be available when you submit a Slurm job")
        print("\n📝 Next steps:")
        print("   1. Submit a test job: sbatch launch_train.sh")
        print("   2. Or start interactive session: srun --partition=h200 --gres=gpu:1g.18gb:1 --pty bash")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
