"""
Download the HuffPost News Category Dataset from Kaggle.
"""
import os
import sys
from pathlib import Path


def download_dataset():
    """Download the dataset using Kaggle API."""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    dataset_file = data_dir / 'News_Category_Dataset_v3.json'
    
    if dataset_file.exists():
        print(f"✓ Dataset already exists at {dataset_file}")
        return True
    
    print("="*60)
    print("DOWNLOADING HUFFPOST NEWS CATEGORY DATASET")
    print("="*60)
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("\n❌ Kaggle package not installed.")
        print("Install it with: pip install kaggle")
        return False
    
    # Check for Kaggle credentials
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        print("\n❌ Kaggle credentials not found!")
        print("\nTo download the dataset, you need to set up Kaggle API:")
        print("\n1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. This will download kaggle.json")
        print(f"5. Move kaggle.json to: {kaggle_json.parent}")
        print("\nOr download manually from:")
        print("https://www.kaggle.com/datasets/rmisra/news-category-dataset")
        return False
    
    # Download dataset
    print("\n📥 Downloading dataset...")
    print("This may take a few minutes...")
    
    try:
        os.system('kaggle datasets download -d rmisra/news-category-dataset -p data --unzip')
        
        if dataset_file.exists():
            print(f"\n✓ Dataset downloaded successfully!")
            print(f"Location: {dataset_file}")
            print(f"Size: {dataset_file.stat().st_size / (1024*1024):.2f} MB")
            return True
        else:
            print("\n❌ Download failed. Please download manually from:")
            print("https://www.kaggle.com/datasets/rmisra/news-category-dataset")
            return False
    
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        return False


if __name__ == "__main__":
    success = download_dataset()
    sys.exit(0 if success else 1)
