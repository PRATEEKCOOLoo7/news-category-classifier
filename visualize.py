"""
Visualization tools for training results and model performance.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd


def plot_training_history(training_log_path='models/trainer_state.json', save_dir='models'):
    """
    Plot training and validation loss/metrics over time.
    
    Args:
        training_log_path (str): Path to trainer state JSON
        save_dir (str): Directory to save plots
    """
    save_dir = Path(save_dir)
    
    # Load training logs
    with open(training_log_path, 'r') as f:
        trainer_state = json.load(f)
    
    log_history = trainer_state['log_history']
    
    # Extract metrics
    train_loss = []
    eval_loss = []
    eval_accuracy = []
    eval_f1 = []
    steps = []
    
    for entry in log_history:
        if 'loss' in entry:
            train_loss.append((entry['step'], entry['loss']))
        if 'eval_loss' in entry:
            steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])
            eval_accuracy.append(entry.get('eval_accuracy', 0))
            eval_f1.append(entry.get('eval_f1_weighted', 0))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    if train_loss:
        train_steps, train_losses = zip(*train_loss)
        axes[0, 0].plot(train_steps, train_losses, 'b-', linewidth=2, label='Training Loss')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    if eval_loss:
        axes[0, 1].plot(steps, eval_loss, 'r-', linewidth=2, label='Validation Loss', marker='o')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Validation Loss Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Validation Accuracy
    if eval_accuracy:
        axes[1, 0].plot(steps, eval_accuracy, 'g-', linewidth=2, label='Accuracy', marker='s')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Validation Accuracy Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Validation F1
    if eval_f1:
        axes[1, 1].plot(steps, eval_f1, 'm-', linewidth=2, label='F1-Score (Weighted)', marker='^')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_title('Validation F1-Score Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = save_dir / 'training_history.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to {output_path}")
    plt.close()


def plot_category_performance(report_path='models/classification_report.json', 
                              save_dir='models', top_n=15):
    """
    Plot per-category F1-scores.
    
    Args:
        report_path (str): Path to classification report JSON
        save_dir (str): Directory to save plots
        top_n (int): Number of top/bottom categories to show
    """
    save_dir = Path(save_dir)
    
    # Load classification report
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    # Extract per-category F1-scores
    categories = []
    f1_scores = []
    
    for category, metrics in report.items():
        if category not in ['accuracy', 'macro avg', 'weighted avg']:
            categories.append(category)
            f1_scores.append(metrics['f1-score'])
    
    # Sort by F1-score
    sorted_data = sorted(zip(categories, f1_scores), key=lambda x: x[1], reverse=True)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Per-Category Performance', fontsize=16, fontweight='bold')
    
    # Top performers
    top_categories, top_scores = zip(*sorted_data[:top_n])
    axes[0].barh(range(len(top_categories)), top_scores, color='green', alpha=0.7)
    axes[0].set_yticks(range(len(top_categories)))
    axes[0].set_yticklabels(top_categories)
    axes[0].set_xlabel('F1-Score')
    axes[0].set_title(f'Top {top_n} Categories by F1-Score')
    axes[0].invert_yaxis()
    axes[0].grid(True, axis='x', alpha=0.3)
    
    # Bottom performers
    bottom_categories, bottom_scores = zip(*sorted_data[-top_n:])
    axes[1].barh(range(len(bottom_categories)), bottom_scores, color='red', alpha=0.7)
    axes[1].set_yticks(range(len(bottom_categories)))
    axes[1].set_yticklabels(bottom_categories)
    axes[1].set_xlabel('F1-Score')
    axes[1].set_title(f'Bottom {top_n} Categories by F1-Score')
    axes[1].invert_yaxis()
    axes[1].grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = save_dir / 'category_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Category performance plot saved to {output_path}")
    plt.close()


def plot_category_distribution(data_path='data/News_Category_Dataset_v3.json',
                               save_dir='models', top_n=20):
    """
    Plot the distribution of categories in the dataset.
    
    Args:
        data_path (str): Path to dataset
        save_dir (str): Directory to save plot
        top_n (int): Number of top categories to show
    """
    save_dir = Path(save_dir)
    
    # Load dataset
    import json
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # Count categories
    category_counts = df['category'].value_counts().head(top_n)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    category_counts.plot(kind='barh', color='skyblue', edgecolor='black')
    plt.xlabel('Number of Articles', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.title(f'Top {top_n} Categories by Article Count', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = save_dir / 'category_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Category distribution plot saved to {output_path}")
    plt.close()


def generate_all_visualizations():
    """Generate all available visualizations."""
    print("="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    models_dir = Path('models')
    
    # Category distribution (always available)
    if Path('data/News_Category_Dataset_v3.json').exists():
        print("\n[1/4] Category distribution...")
        try:
            plot_category_distribution()
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
    
    # Training history (requires training)
    if (models_dir / 'trainer_state.json').exists():
        print("\n[2/4] Training history...")
        try:
            plot_training_history()
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
    else:
        print("\n[2/4] Training history... ⊘ (train model first)")
    
    # Category performance (requires training)
    if (models_dir / 'classification_report.json').exists():
        print("\n[3/4] Category performance...")
        try:
            plot_category_performance()
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
    else:
        print("\n[3/4] Category performance... ⊘ (train model first)")
    
    # Confusion matrix (auto-generated during training)
    if (models_dir / 'confusion_matrix.png').exists():
        print("\n[4/4] Confusion matrix... ✓ (already exists)")
    else:
        print("\n[4/4] Confusion matrix... ⊘ (generated during training)")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    generate_all_visualizations()
