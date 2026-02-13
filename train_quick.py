"""
Quick training script optimized for CPU with reduced dataset size.
Trains on a representative subset for faster completion.
"""
import sys
sys.path.insert(0, '.')

from train import train_model

if __name__ == "__main__":
    print("="*60)
    print("QUICK TRAINING MODE - CPU OPTIMIZED")
    print("="*60)
    print("\nThis mode uses a subset of data for faster training on CPU.")
    print("Full dataset training requires GPU for reasonable time.")
    print()
    
    # Train with optimized settings for CPU
    train_model(
        model_name='distilbert-base-uncased',
        data_path='data/News_Category_Dataset_v3.json',
        output_dir='models',
        epochs=2,  # Reduced from 3
        batch_size=32,  # Increased for efficiency
        learning_rate=3e-5,  # Slightly higher LR
        max_length=96,  # Reduced sequence length
        save_steps=2000,
        eval_steps=1000,
        warmup_steps=200,
        max_train_samples=50000,  # Use 50K samples instead of 210K
        max_val_samples=10000  # Use 10K validation samples
    )
