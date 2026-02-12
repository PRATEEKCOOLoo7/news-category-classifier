"""
Quick setup script to verify dependencies and environment.
"""
import sys
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("  ❌ Python 3.8+ required")
        return False
    print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_dependencies():
    """Check if all required packages are installed."""
    print("\nChecking dependencies...")
    
    required = {
        'transformers': 'transformers',
        'torch': 'torch',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'tqdm': 'tqdm',
        'kaggle': 'kaggle'
    }
    
    missing = []
    for module, package in required.items():
        try:
            importlib.import_module(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def check_project_structure():
    """Check if project structure is correct."""
    print("\nChecking project structure...")
    
    required_dirs = ['data', 'models', 'src']
    required_files = [
        'src/__init__.py',
        'src/data_loader.py',
        'src/preprocessor.py',
        'src/model.py',
        'train.py',
        'predict.py',
        'requirements.txt'
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ missing")
            all_good = False
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ❌ {file_name} missing")
            all_good = False
    
    return all_good


def check_dataset():
    """Check if dataset is downloaded."""
    print("\nChecking dataset...")
    
    dataset_path = Path('data/News_Category_Dataset_v3.json')
    if dataset_path.exists():
        size_mb = dataset_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Dataset found ({size_mb:.2f} MB)")
        return True
    else:
        print("  ⚠ Dataset not found")
        print("    Run: python download_dataset.py")
        print("    Or download from: https://www.kaggle.com/datasets/rmisra/news-category-dataset")
        return False


def check_gpu():
    """Check if GPU is available."""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ GPU available: {gpu_name}")
            print(f"    This will significantly speed up training!")
            return True
        else:
            print("  ⚠ No GPU detected")
            print("    Training will use CPU (slower)")
            return False
    except:
        print("  ⚠ Could not check GPU")
        return False


def main():
    """Run all verification checks."""
    print("="*60)
    print("NEWS CATEGORY CLASSIFIER - SETUP VERIFICATION")
    print("="*60)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_project_structure(),
        check_dataset(),
    ]
    
    check_gpu()  # Optional check
    
    print("\n" + "="*60)
    if all(checks):
        print("✅ ALL CHECKS PASSED - READY TO TRAIN!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Ensure dataset is downloaded: python download_dataset.py")
        print("  2. Train the model: python train.py")
        print("  3. Make predictions: python predict.py --text \"Your text here\"")
        return 0
    else:
        print("⚠ SOME CHECKS FAILED - FIX ISSUES ABOVE")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
