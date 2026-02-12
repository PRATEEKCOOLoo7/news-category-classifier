"""
Main training script for news category classification.
"""
import os
import argparse
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from src.data_loader import load_dataset
from src.preprocessor import NewsDataset


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        dict: Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def plot_confusion_matrix(y_true, y_pred, label_encoder, save_path='models/confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_encoder: Label encoder to get category names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, 
        annot=False,  # Too many classes for annotations
        fmt='d',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('Confusion Matrix - News Category Classification', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()


def train_model(
    model_name='distilbert-base-uncased',
    data_path='data/News_Category_Dataset_v3.json',
    output_dir='models',
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    max_length=128,
    save_steps=1000,
    eval_steps=500,
    warmup_steps=500
):
    """
    Train the news classification model.
    
    Args:
        model_name (str): Pretrained model name
        data_path (str): Path to dataset
        output_dir (str): Directory to save model
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        max_length (int): Maximum sequence length
        save_steps (int): Save checkpoint every N steps
        eval_steps (int): Evaluate every N steps
        warmup_steps (int): Number of warmup steps
    """
    print("="*60)
    print("NEWS CATEGORY CLASSIFICATION - TRAINING")
    print("="*60)
    
    # Load and prepare data
    print("\n[1/6] Loading dataset...")
    train_df, val_df, label_encoder, num_classes = load_dataset(data_path)
    
    # Initialize tokenizer
    print(f"\n[2/6] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    print("\n[3/6] Creating datasets...")
    train_dataset = NewsDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = NewsDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Initialize model
    print(f"\n[4/6] Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Training arguments
    print(f"\n[5/6] Configuring training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy='steps',
        eval_steps=eval_steps,
        save_strategy='steps',
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model='f1_weighted',
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
        report_to='none',  # Disable wandb/tensorboard
        remove_unused_columns=False
    )
    
    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max sequence length: {max_length}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print(f"\n[6/6] Starting training...")
    print("-"*60)
    train_result = trainer.train()
    
    # Save final model
    print("\nSaving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    eval_results = trainer.evaluate()
    
    print(f"\nValidation Metrics:")
    print(f"  Accuracy:     {eval_results['eval_accuracy']:.4f}")
    print(f"  F1 (Macro):   {eval_results['eval_f1_macro']:.4f}")
    print(f"  F1 (Weighted): {eval_results['eval_f1_weighted']:.4f}")
    
    # Get predictions for confusion matrix
    print("\nGenerating confusion matrix...")
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = val_df['label'].values
    
    # Save confusion matrix
    plot_confusion_matrix(y_true, y_pred, label_encoder)
    
    # Save detailed classification report
    print("\nGenerating classification report...")
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    report_path = Path(output_dir) / 'classification_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Classification report saved to {report_path}")
    
    # Print top and bottom performing categories
    category_f1 = {cat: report[cat]['f1-score'] for cat in label_encoder.classes_}
    sorted_categories = sorted(category_f1.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 performing categories:")
    for cat, f1 in sorted_categories[:5]:
        print(f"  {cat:20s} F1: {f1:.4f}")
    
    print("\nBottom 5 performing categories:")
    for cat, f1 in sorted_categories[-5:]:
        print(f"  {cat:20s} F1: {f1:.4f}")
    
    # Save training summary
    summary = {
        'model_name': model_name,
        'num_classes': num_classes,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_length': max_length,
        'accuracy': float(eval_results['eval_accuracy']),
        'f1_macro': float(eval_results['eval_f1_macro']),
        'f1_weighted': float(eval_results['eval_f1_weighted'])
    }
    
    summary_path = Path(output_dir) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Training summary saved to {summary_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {output_dir}")
    print(f"To make predictions, use: python predict.py --text \"Your news headline here\"")
    
    return trainer, eval_results


def main():
    parser = argparse.ArgumentParser(description='Train news category classifier')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                        help='Pretrained model name')
    parser.add_argument('--data_path', type=str, 
                        default='data/News_Category_Dataset_v3.json',
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    train_model(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()
